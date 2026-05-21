# Recursive Prerequisite Knowledge Tracer

An AI-powered educational tool implementing the research paper "Learning What You Don't Know: Recursive Prerequisite Knowledge Tracing in Conversational AI Tutors for Efficient Learning". The system helps learners discover knowledge gaps by recursively tracing prerequisite concepts for any academic topic.

## System Design and Architecture

### Core Principle
The system addresses the fundamental learning challenge: "learners often don't know what they don't know." Instead of assuming foundational knowledge, it systematically identifies and traces prerequisite concepts until reaching the learner's actual knowledge boundary.

### Architecture Components

#### 1. **Knowledge Tracer Engine** (`knowledge_tracer.py`)
- **Dynamic Prerequisite Extraction**: Uses GPT-4o to identify directly relevant technical prerequisites
- **Recursive Analysis**: Automatically expands unknown concepts to deeper levels
- **Fundamental Concept Detection**: Stops recursion at basic mathematical/logical concepts
- **Adaptive Explanation Generation**: Creates personalized explanations based on known/unknown concepts

#### 2. **Interactive User Interface** (`app.py`)
- **Question Analysis Module**: Provides initial topic explanation and importance
- **Interactive Assessment Cards**: Binary knowledge checking (✅ know / ❌ don't know)
- **Real-time Expansion**: Immediately expands unknown concepts inline
- **Multi-view Visualization**: Knowledge check, dependency graph, and learning path views

#### 3. **Session Management System**
- **Knowledge State Tracking**: Maintains user responses across recursive levels
- **Duplicate Concept Handling**: Consistent status for concepts appearing multiple times
- **Expansion State Management**: Tracks which concepts have been expanded

## Key Features and Innovations

### 1. **Dynamic Knowledge Graph Generation**
- No pre-stored knowledge base required
- Works across all academic domains
- Identifies only directly relevant prerequisites (not broad foundational skills)
- Example: For "backpropagation" → "gradient descent, chain rule, loss functions" (NOT "basic math")

### 2. **Recursive Prerequisite Tracing**
- **Depth Control**: Configurable recursion levels (L0 to L3 default)
- **Intelligent Stopping**: Halts at fundamental concepts or max depth
- **Context-Aware**: Adjusts based on education level (Elementary to PhD)

### 3. **Interactive Knowledge Assessment**
- **Binary Decision Interface**: Simple ✅/❌ buttons for rapid assessment
- **Immediate Expansion**: Unknown concepts automatically reveal sub-prerequisites
- **Visual Feedback**: Dimmed display for known concepts, highlighted expansion for unknown
- **Consistent State Management**: Same concept shows identical status across all appearances

### 4. **Multi-Modal Learning Support**
- **Knowledge Dependency Graph**: Visual network showing concept relationships
- **Hierarchical Learning Path**: Tree structure displaying optimal learning sequence
- **Comprehensive Explanations**: AI-generated content addressing all identified knowledge gaps

## System Flow and Methodology

### Phase 1: Question Analysis
1. User inputs academic question/topic
2. GPT-4o analyzes topic for:
   - Clear conceptual explanation
   - Learning importance and relevance
   - Key technical concepts involved
3. System displays analysis with identified key concepts

### Phase 2: Interactive Knowledge Tracing
1. **Level 1 Assessment**: Present key concepts as prerequisite cards
2. **User Evaluation**: Binary choice for each concept (know/don't know)
3. **Recursive Expansion**: For unknown concepts:
   - Extract their prerequisites using GPT-4o
   - Create new assessment cards at Level 2
   - Continue recursion until knowledge boundary found

### Phase 3: Learning Path Generation
1. **Dependency Analysis**: Build hierarchical relationship tree
2. **Path Optimization**: Sequence concepts from fundamental to advanced
3. **Personalized Explanation**: Generate comprehensive explanation addressing all knowledge gaps

## Technical Implementation

### GPT-4o Integration
- **Model**: OpenAI GPT-4o with JSON structured output
- **Temperature**: 0.2-0.3 for consistent prerequisite extraction
- **Prompt Engineering**: Focused on direct technical dependencies
- **Error Handling**: Robust parsing for various response formats

### State Management
- **Session Persistence**: Maintains state across user interactions
- **Real-time Updates**: Immediate UI refresh on user actions
- **Conflict Resolution**: Handles duplicate concepts across expansion levels

### Visualization Components
- **Interactive Graph**: Plotly-based network visualization with level indicators
- **Tree Display**: ASCII-art style hierarchical prerequisite tree
- **Responsive Cards**: Dynamic styling based on knowledge status

## Evaluation Metrics and Features

### Learning Efficiency Indicators
- **Prerequisite Discovery**: Number of unknown concepts identified
- **Recursion Depth**: Maximum levels traced per learning session
- **Knowledge Coverage**: Percentage of prerequisites assessed
- **Time to Boundary**: Steps required to reach knowledge limits

### Adaptive Personalization
- **Education Level Adjustment**: Content complexity based on learner background
- **Context-Aware Prerequisites**: Domain-specific concept identification
- **Progressive Disclosure**: Reveals complexity incrementally

### User Experience Design
- **Cognitive Load Reduction**: Simple binary decisions instead of complex assessments
- **Visual Hierarchy**: Clear level indicators and dependency relationships
- **Immediate Feedback**: Real-time expansion and status updates

## Detailed Implementation

### Core Algorithm: Recursive Prerequisite Extraction

```python
def extract_prerequisites(self, topic: str, context: str = "") -> List[Prerequisite]:
    """
    Extracts directly relevant prerequisites using GPT-4o
    
    Key Design Decisions:
    - Temperature: 0.2 (low for consistency)
    - Focus: Technical concepts only, not general skills  
    - Depth: 2-4 concepts per level (manageable cognitive load)
    - Format: Structured JSON with explanations and relevance
    """
```

### Interactive Assessment Logic

```python
def display_prerequisite_card(prereq: Prerequisite, key_prefix: str):
    """
    Renders interactive knowledge assessment cards
    
    Features:
    - Binary choice interface (✅/❌ buttons)
    - Immediate visual feedback (dimming for known concepts)
    - Duplicate concept handling (shows "Already confirmed" status)
    - Real-time expansion of unknown concepts
    - Level indicators [L0], [L1], [L2], etc.
    """
```

### Recursion Control Mechanism

```python
def expand_unknown_prerequisites(self, prerequisites: List[Prerequisite], current_depth: int):
    """
    Controls recursive expansion with intelligent stopping
    
    Stopping Conditions:
    1. Maximum depth reached (user-configurable: L0-L3 default)
    2. Fundamental concepts detected (basic arithmetic, algebra)
    3. All concepts marked as "known" by user
    4. No prerequisites available for concept
    """
```

### State Management Architecture

```python
# Session State Variables
st.session_state.knowledge_status = {}      # Dict[concept_name, bool] 
st.session_state.expanded_unknowns = set()  # Set of expanded concept names
st.session_state.current_tree = None        # Prerequisite tree structure
st.session_state.question_analysis = None   # Initial topic analysis
```

### GPT-4o Prompt Engineering

#### 1. Prerequisite Extraction Prompt
```
For the topic: "{topic}"
Education level: {education_level}

Identify DIRECTLY RELEVANT technical concepts essential for understanding this topic.
Focus on immediate conceptual dependencies, not broad foundational skills.

Example: For "backpropagation" → gradient descent, chain rule, loss functions
NOT: basic math, computer literacy, programming basics

Return 2-4 critical prerequisites as JSON...
```

#### 2. Comprehensive Explanation Prompt
```
Explain: "{topic}"

Known: {known_prerequisites}
Unknown: {unknown_prerequisites}

Provide explanation that:
1. THOROUGHLY explains unknown concepts first
2. Builds from known to unknown concepts
3. Uses step-by-step progression
4. Includes examples for each unknown concept
```

### Visualization Implementation

#### Knowledge Dependency Graph
- **Library**: Plotly with NetworkX backend
- **Node Properties**: Size decreases with depth, color indicates status
- **Layout Algorithm**: Spring layout for optimal positioning  
- **Interactive Features**: Hover tooltips showing concept details

#### Hierarchical Learning Path
- **Format**: ASCII tree structure using Unicode characters
- **Symbols**: ├── └── │ for tree branches
- **Status Indicators**: ❌ (unknown), ✅ (known), ❓ (unchecked)
- **Level Display**: (Level N) annotations

## File Structure and Components

```
Education_AI/
├── app.py                    # Main Streamlit application
├── knowledge_tracer.py       # Core algorithm implementation  
├── requirements.txt          # Python dependencies
├── .env                     # OpenAI API configuration
├── education_ai_env/        # Virtual environment
├── system_flowchart.md      # System flow documentation
└── README.md               # This documentation
```

### Key Functions by Module

#### `knowledge_tracer.py`
- `analyze_question()`: Initial topic analysis with GPT-4o
- `extract_prerequisites()`: Recursive prerequisite identification
- `expand_unknown_prerequisites()`: Manages recursive expansion
- `generate_comprehensive_explanation()`: Creates personalized tutorials
- `check_fundamental()`: Detects stopping conditions

#### `app.py`  
- `display_prerequisite_card()`: Interactive assessment interface
- `create_knowledge_graph()`: Plotly visualization generation
- `display_nested_learning_path()`: Hierarchical path rendering
- `main()`: Application orchestration and session management

## Performance and Scalability

### API Usage Optimization
- **Caching**: Session-level prerequisite caching to avoid redundant calls
- **Batch Processing**: Group related concepts for efficient extraction
- **Error Handling**: Graceful fallbacks for API failures
- **Rate Limiting**: Controlled recursion depth to manage API usage

### Response Time Targets
- Initial question analysis: < 3 seconds
- Prerequisite extraction per level: < 2 seconds  
- Total session completion: < 30 seconds (typical 3-level recursion)

### Memory Management
- Streamlit session state for user data persistence
- Garbage collection of unused prerequisite trees
- Efficient JSON parsing for GPT-4o responses

## System Requirements and Constraints

### Technical Requirements
- **Python Environment**: Python 3.9+ with virtual environment support
- **API Access**: OpenAI GPT-4o API key with sufficient quota
- **Dependencies**: Streamlit, OpenAI Python SDK, NetworkX, Plotly
- **Browser Compatibility**: Modern web browsers supporting HTML5 and JavaScript
- **Memory**: Minimum 2GB RAM for local deployment

### Design Constraints and Assumptions
- **Language Limitation**: Currently supports English-language academic content only
- **Domain Scope**: Optimized for STEM and technical subjects with clear prerequisite hierarchies
- **Recursion Depth**: Maximum 6 levels to prevent infinite loops and manage cognitive load
- **Concept Granularity**: Focuses on conceptual knowledge, not procedural skills
- **Assessment Validity**: Assumes users can accurately self-assess their knowledge

### Comparison with Existing Approaches

| Feature | Traditional Tutoring | Adaptive Learning Systems | Our Approach |
|---------|---------------------|---------------------------|--------------|
| Knowledge Gap Detection | Manual assessment | Pre-defined skill trees | Dynamic discovery |
| Prerequisite Mapping | Expert-created | Static knowledge graphs | AI-generated |
| Personalization | Human intuition | Algorithmic adaptation | Interactive recursion |
| Domain Coverage | Limited by tutor expertise | Curriculum-specific | Universal (any topic) |
| Scalability | One-to-one limitation | High but rigid | High and flexible |

## Evaluation Framework

### Quantitative Metrics
1. **Discovery Efficiency**: Number of unknown concepts identified per session
2. **Recursion Effectiveness**: Average depth reached before knowledge boundary
3. **Time to Completion**: Total session duration from question to explanation
4. **Concept Coverage**: Percentage of relevant prerequisites identified
5. **User Engagement**: Click-through rates and session completion rates

### Qualitative Assessment Criteria
1. **Prerequisite Relevance**: Expert evaluation of identified dependencies
2. **Explanation Quality**: Comprehensiveness and clarity of generated tutorials
3. **User Experience**: Cognitive load and interface usability
4. **Learning Effectiveness**: Post-session comprehension improvements

### Validation Methodology
- **Expert Review**: Domain experts validate prerequisite accuracy
- **User Studies**: Comparative effectiveness against traditional methods
- **A/B Testing**: Interface variants and recursion strategies
- **Long-term Tracking**: Learning outcome measurements over time

## Setup

1. Create and activate virtual environment:

```bash
python3 -m venv education_ai_env
source education_ai_env/bin/activate  # On Mac/Linux
# or
education_ai_env\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your_actual_api_key_here
```

4. Run the app:

**Standard run (foreground):**

```bash
streamlit run app.py
```

**Run in background (keeps running after terminal closes):**

```bash
source education_ai_env/bin/activate && nohup streamlit run app.py > streamlit.log 2>&1 &
```

**Stop the app:**

```bash
pkill -f streamlit
```

**Check if app is running:**

```bash
ps aux | grep streamlit | grep -v grep
```

## How It Works

1. **Ask a Question**: Enter any academic topic or question
2. **Check Prerequisites**: The system shows prerequisite concepts with explanations
3. **Mark Your Knowledge**: Click "I know this" or "I don't know" for each concept
4. **Recursive Analysis**: For unknown concepts, the system finds their prerequisites
5. **Get Your Path**: Receive a personalized learning sequence based on gaps

## Architecture

- `knowledge_tracer.py`: Core logic for prerequisite extraction and recursive tracing
- `app.py`: Streamlit interface with interactive components
- Uses GPT-4o for intelligent prerequisite identification
- Session state management for tracking user progress

## Example Use Cases

- Students preparing for advanced topics
- Self-learners identifying knowledge gaps
- Educators designing curriculum paths
- Anyone wanting to learn efficiently by understanding foundations first

## Demo Questions

- How does backpropagation work in neural networks?
- Explain how dynamic programming works in algorithms
