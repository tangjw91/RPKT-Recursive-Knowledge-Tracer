import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Prerequisite:
    name: str
    explanation: str
    relevance: str
    example: Optional[str] = None
    depth_level: int = 0
    user_knows: Optional[bool] = None
    prerequisites: List['Prerequisite'] = field(default_factory=list)

class KnowledgeTracer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your .env file")
        self.client = OpenAI(api_key=api_key)
        self.max_depth = 5
        self.current_tree = None
        
    def extract_prerequisites(self, topic: str, context: str = "") -> List[Prerequisite]:
        prompt = f"""
        For the topic/question: "{topic}"
        Context: {context if context else "General academic learning"}
        
        Identify the DIRECTLY RELEVANT technical concepts that are essential prerequisites for understanding this specific topic.
        Focus on the immediate conceptual dependencies, not broad foundational skills.
        
        For example:
        - For "backpropagation in neural networks", prerequisites would be: gradient descent, chain rule, loss functions, forward propagation
        - NOT: basic math, computer literacy, programming basics
        
        Return 2-4 most critical prerequisites that are:
        1. Directly used in understanding the main topic
        2. Technical concepts, not general skills
        3. One level below the main topic in complexity
        
        For each prerequisite, provide:
        1. Name: Specific technical concept name
        2. Explanation: 2-3 sentence explanation of the concept
        3. Relevance: How this concept is specifically used in the main topic
        4. Example: A brief concrete example (optional)
        
        Return as JSON with the key "prerequisites" containing an array:
        {{
            "prerequisites": [
                {{
                    "name": "Concept Name",
                    "explanation": "Clear explanation...",
                    "relevance": "This is directly used in [main topic] for...",
                    "example": "For instance..."
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert educator who identifies DIRECT technical prerequisites. Focus on concepts that are immediately relevant and directly used in the main topic, not general foundational skills."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            prerequisites = []
            
            # Handle different response formats
            if isinstance(data, list):
                prereq_list = data
            elif isinstance(data, dict):
                if "prerequisites" in data:
                    prereq_list = data["prerequisites"]
                elif "concepts" in data:
                    prereq_list = data["concepts"]
                else:
                    # Try to find the first list value
                    for value in data.values():
                        if isinstance(value, list):
                            prereq_list = value
                            break
                    else:
                        prereq_list = []
            else:
                prereq_list = []
                
            for item in prereq_list:
                if isinstance(item, dict):
                    prerequisites.append(Prerequisite(
                        name=item.get("name", ""),
                        explanation=item.get("explanation", ""),
                        relevance=item.get("relevance", ""),
                        example=item.get("example")
                    ))
                elif isinstance(item, str):
                    # Handle simple string items
                    prerequisites.append(Prerequisite(
                        name=item,
                        explanation=f"Understanding {item}",
                        relevance=f"Required prerequisite concept",
                        example=None
                    ))
                
            return prerequisites
            
        except Exception as e:
            print(f"Error extracting prerequisites: {e}")
            return []
    
    def check_fundamental(self, concept: str) -> bool:
        fundamental_concepts = [
            "basic arithmetic", "addition", "subtraction", "multiplication", "division",
            "variables", "functions", "basic algebra", "numbers", "equations"
        ]
        return any(fundamental in concept.lower() for fundamental in fundamental_concepts)
    
    def trace_prerequisites_recursive(self, topic: str, depth: int = 0, context: str = "") -> Prerequisite:
        if depth >= self.max_depth:
            return Prerequisite(
                name=topic,
                explanation="Maximum depth reached",
                relevance="Fundamental concept",
                depth_level=depth
            )
        
        root = Prerequisite(
            name=topic,
            explanation=f"Main topic: {topic}",
            relevance="This is what you want to learn",
            depth_level=depth
        )
        
        if self.check_fundamental(topic):
            return root
            
        prerequisites = self.extract_prerequisites(topic, context)
        root.prerequisites = prerequisites
        
        for prereq in prerequisites:
            prereq.depth_level = depth + 1
            
        return root
    
    def expand_unknown_prerequisites(self, prerequisites: List[Prerequisite], current_depth: int) -> None:
        for prereq in prerequisites:
            # Check if this prerequisite is marked as unknown and hasn't been expanded yet
            if prereq.user_knows == False and not prereq.prerequisites:
                # Only expand if we haven't reached max depth
                next_depth = current_depth + 1
                if next_depth < self.max_depth and not self.check_fundamental(prereq.name):
                    prereq.prerequisites = self.extract_prerequisites(prereq.name)
                    for sub_prereq in prereq.prerequisites:
                        sub_prereq.depth_level = next_depth
            # Recursively check already expanded prerequisites
            elif prereq.prerequisites:
                self.expand_unknown_prerequisites(prereq.prerequisites, prereq.prerequisites[0].depth_level if prereq.prerequisites else current_depth + 1)
    
    def analyze_question(self, question: str, education_level: str) -> Dict[str, any]:
        prompt = f"""
        Analyze this question/topic: "{question}"
        Education level: {education_level}
        
        Provide:
        1. A clear explanation of what this topic/concept is about (3-4 sentences)
        2. Why it's important to learn
        3. The key technical concepts that are directly involved in this topic (return as detailed prerequisite objects)
        
        For example, for "How does backpropagation work in neural networks?":
        - Key concepts would be: forward propagation, gradient descent, chain rule, loss functions
        - NOT: basic programming, math fundamentals, computer basics
        
        Return as JSON with format:
        {{
            "explanation": "Clear explanation of the topic...",
            "importance": "Why this is important to learn...",
            "key_concepts": [
                {{
                    "name": "Concept name",
                    "explanation": "What this concept is (2-3 sentences)",
                    "relevance": "How it's used in the main topic",
                    "example": "Brief example (optional)"
                }},
                ...
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert educator who analyzes learning topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "explanation": f"Error analyzing question: {e}",
                "importance": "",
                "key_concepts": []
            }
    
    def generate_comprehensive_explanation(self, topic: str, known_prerequisites: List[str], unknown_prerequisites: List[str]) -> str:
        prompt = f"""
        Explain the topic: "{topic}"
        
        The learner already understands these prerequisites: {', '.join(known_prerequisites) if known_prerequisites else 'None'}
        
        The learner DOES NOT understand these concepts: {', '.join(unknown_prerequisites) if unknown_prerequisites else 'None'}
        
        Provide a comprehensive explanation that:
        1. Builds on the known prerequisites
        2. THOROUGHLY explains all the unknown concepts first before using them
        3. For each unknown concept, provide clear explanations with examples
        4. Then connects everything to explain the main topic
        5. Uses step-by-step progression from what they know to what they need to learn
        
        Start by explaining the unknown prerequisites in detail, then show how they apply to the main topic.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an excellent teacher who fills knowledge gaps by thoroughly explaining unknown concepts before using them."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    def generate_explanation(self, topic: str, known_prerequisites: List[str]) -> str:
        prompt = f"""
        Explain the topic: "{topic}"
        
        The learner already understands these prerequisites: {', '.join(known_prerequisites)}
        
        Provide a clear, comprehensive explanation that:
        1. Builds on the known prerequisites
        2. Introduces the new concept step by step
        3. Uses examples and analogies
        4. Highlights key insights
        
        Keep the explanation engaging and accessible.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an excellent teacher who explains complex topics clearly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    def get_learning_path(self, root: Prerequisite) -> List[Tuple[str, int]]:
        path = []
        
        def traverse(node: Prerequisite):
            if node.user_knows == False or node.user_knows is None:
                for prereq in node.prerequisites:
                    traverse(prereq)
                path.append((node.name, node.depth_level))
        
        traverse(root)
        return path