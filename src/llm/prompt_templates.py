"""
Prompt Templates for LLM Heuristic Generator

DRA-themed prompts for generating optimization operators.
Uses DRA metaphors: Prophet (DRL agent), Congregation (population), Divine Speech (LLM).
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A template for generating LLM prompts."""

    name: str
    description: str
    template: str
    operator_type: str  # mutation, crossover, local_search, diversification


# Base system prompt for the LLM - Comprehensive English version
SYSTEM_PROMPT = """You are an expert optimization algorithm designer specializing in metaheuristic operators for feature selection problems. Your task is to generate novel and effective optimization operators that will be dynamically integrated into a running Divine Religions Algorithm (DRA).

## CONTEXT
You are the "Divine Speech" (İlahi Kelam) component in a hyper-heuristic framework. When the optimization process stagnates, a DRL agent (the "Prophet") requests new operators from you to help the population escape local optima or improve convergence.

## STRICT RULES
1. **Output ONLY Python code** - No explanations, comments outside code, or markdown text
2. **Function name MUST be exactly `custom_operator`**
3. **Function signature MUST be:**
   ```python
   def custom_operator(position, leader, iteration, max_iteration):
   ```
   - `position`: np.ndarray - Current individual's continuous representation (shape: (n_features,))
   - `leader`: np.ndarray - Best solution found so far (shape: (n_features,))
   - `iteration`: int - Current iteration number
   - `max_iteration`: int - Maximum iterations
   
4. **Function MUST return:** np.ndarray of same shape as position (new continuous values)

5. **Allowed imports:** Only NumPy (already imported as `np`)
   - Use: np.random, np.sin, np.cos, np.exp, np.abs, np.clip, np.mean, np.std, etc.

6. **FORBIDDEN (will cause rejection):**
   - File I/O operations (open, read, write)
   - Network calls (requests, urllib, socket)
   - System commands (os.system, subprocess)
   - Dynamic imports (__import__, importlib)
   - eval() or exec()
   - Any side effects outside the function

7. **Output value constraints:**
   - Return values should be in range [-10, 10]
   - Avoid NaN or Inf values
   - Handle edge cases gracefully

8. **Code quality:**
   - Keep it simple and efficient
   - Use vectorized NumPy operations when possible
   - Avoid loops when vectorization is possible

## EXAMPLE OUTPUT FORMAT
```python
def custom_operator(position, leader, iteration, max_iteration):
    progress = iteration / max_iteration
    # Adaptive exploration: large changes early, small refinements late
    exploration_rate = 1.0 - progress
    noise = np.random.randn(len(position)) * exploration_rate
    # Move toward leader with decreasing randomness
    new_position = position + 0.5 * (leader - position) * progress + noise * 0.3
    return np.clip(new_position, -10, 10)
```

## OPERATOR DESIGN PRINCIPLES
- **Exploration operators**: Should create large, diverse perturbations to escape local optima
- **Exploitation operators**: Should make small, directed movements toward promising regions
- **Balance**: Good operators often combine both aspects adaptively based on progress
- **Novelty**: Try creative mathematical formulations (trigonometric, exponential, probability distributions)

Remember: Output ONLY the Python function code, nothing else. Use /no_think mode.
"""


# Mutation Operators - Detailed English Templates
MUTATION_TEMPLATES = [
    PromptTemplate(
        name="adaptive_mutation",
        description="Progress-based adaptive mutation with dynamic intensity",
        operator_type="mutation",
        template="""Design an ADAPTIVE MUTATION operator for feature selection optimization.

REQUIREMENTS:
- Mutation intensity should DECREASE as optimization progresses
- Early iterations (progress < 0.3): Large mutations for exploration (high variance noise)
- Middle iterations (0.3 < progress < 0.7): Moderate mutations balancing exploration/exploitation
- Late iterations (progress > 0.7): Small, fine-tuning mutations for exploitation
- Include stochastic elements (use np.random)
- Consider the leader position for guidance but don't converge too quickly

OPTIMIZATION STATE:
- Current iteration: {iteration} / {max_iteration}
- Progress ratio: {progress:.1%}
- Population diversity: {diversity:.3f} (0=identical, 1=maximally diverse)
- Recent fitness improvement: {improvement:.5f}

DESIGN HINT: Consider using exponential decay for mutation rate, or trigonometric functions for oscillating exploration patterns.
""",
    ),
    PromptTemplate(
        name="elite_guided_mutation",
        description="Leader-guided mutation maintaining diversity",
        operator_type="mutation",
        template="""Design a LEADER-GUIDED MUTATION operator that balances exploitation with diversity preservation.

REQUIREMENTS:
- Pull the current position toward the leader (best solution)
- BUT avoid complete convergence - maintain population diversity
- The attraction strength should be proportional to the fitness gap
- Add controlled randomness to prevent premature convergence
- With {diversity:.1%} current diversity, adjust exploration accordingly

OPTIMIZATION STATE:
- Average distance from leader: {mean_cost:.4f}
- Is stagnating: {is_stagnating}
- Current diversity level: {diversity:.3f}

DESIGN HINT: Use a combination of directional movement (toward leader) and orthogonal perturbation (perpendicular to the leader direction) to explore while exploiting.
""",
    ),
    PromptTemplate(
        name="random_reset_mutation",
        description="Partial random reset for escaping local optima",
        operator_type="mutation",
        template="""Design a PARTIAL RANDOM RESET operator for feature selection.

REQUIREMENTS:
- Randomly reinitialize a SUBSET of dimensions (not all)
- Reset probability should DECREASE as optimization progresses
- Preferentially preserve dimensions where position matches leader (likely good features)
- Reinitialize other dimensions with uniform random values in [-10, 10]

OPTIMIZATION STATE:
- Problem dimensionality: {dim} features
- Currently selected features: {num_selected}
- Progress: {progress:.1%}

DESIGN HINT: Create a probability mask where each dimension has a chance of being reset. Dimensions closer to leader values should have lower reset probability.
""",
    ),
    PromptTemplate(
        name="gaussian_mutation",
        description="Gaussian noise mutation with adaptive variance",
        operator_type="mutation",
        template="""Design a GAUSSIAN MUTATION operator with adaptive variance.

REQUIREMENTS:
- Add Gaussian noise to each dimension
- Variance should adapt based on:
  1. Optimization progress (decrease over time)
  2. Stagnation status (increase if stagnating)
  3. Distance from leader (larger for far individuals)
- Ensure output stays within bounds [-10, 10]

OPTIMIZATION STATE:
- Iteration: {iteration}/{max_iteration}
- Is stagnating: {is_stagnating}
- Stagnation count: {stagnation_count} iterations without improvement

DESIGN HINT: Base variance = (1 - progress) * base_sigma. If stagnating, multiply by stagnation_factor > 1.
""",
    ),
]


# Crossover Operators - Detailed English Templates
CROSSOVER_TEMPLATES = [
    PromptTemplate(
        name="blend_crossover",
        description="BLX-alpha style blend crossover with leader",
        operator_type="crossover",
        template="""Design a BLEND CROSSOVER (BLX-alpha) operator combining current position with the leader.

REQUIREMENTS:
- Create offspring in the region between position and leader, extended by alpha
- For each dimension: offspring = position + beta * (leader - position)
  where beta is randomly sampled from [-alpha, 1+alpha]
- Common alpha values: 0.3 to 0.5
- Adapt alpha based on optimization progress

OPTIMIZATION STATE:
- Progress: {progress:.1%}
- Current diversity: {diversity:.3f}

DESIGN HINT: Early stage: larger alpha for exploration. Late stage: smaller alpha focused near parents.
""",
    ),
    PromptTemplate(
        name="arithmetic_crossover",
        description="Weighted arithmetic combination with random weights",
        operator_type="crossover",
        template="""Design an ARITHMETIC CROSSOVER operator with adaptive weighting.

REQUIREMENTS:
- Combine position and leader: offspring = w * position + (1-w) * leader
- Weight w should be randomly generated but biased based on:
  1. Progress (favor leader more as optimization advances)
  2. Quality difference (better parent gets higher weight)
- Add small "mystical" perturbation for novelty (exploration noise)

OPTIMIZATION STATE:
- Progress: {progress:.1%}
- Mean population cost: {mean_cost:.4f}

DESIGN HINT: Use beta distribution for weight sampling - Beta(alpha, beta) where parameters depend on progress.
""",
    ),
    PromptTemplate(
        name="simulated_binary_crossover",
        description="SBX-style crossover adapted for continuous representation",
        operator_type="crossover",
        template="""Design a SIMULATED BINARY CROSSOVER (SBX) operator.

REQUIREMENTS:
- Implement the SBX formula with distribution index eta
- For each dimension, generate beta from the SBX distribution
- offspring = 0.5 * ((position + leader) + beta * (leader - position))
- Eta controls spread: low eta = wide spread, high eta = near parents
- Adapt eta based on optimization stage

OPTIMIZATION STATE:
- Progress: {progress:.1%}
- Is stagnating: {is_stagnating}

DESIGN HINT: eta = 2 + 20 * progress (start exploratory, end exploitative)
""",
    ),
]


# Local Search Operators - Detailed English Templates
LOCAL_SEARCH_TEMPLATES = [
    PromptTemplate(
        name="neighborhood_search",
        description="Local neighborhood exploration with adaptive step size",
        operator_type="local_search",
        template="""Design a LOCAL NEIGHBORHOOD SEARCH operator for fine-tuning solutions.

REQUIREMENTS:
- Explore the immediate neighborhood of current position
- Step size should be SMALL and DECREASE with progress
- Direction should be influenced by the leader position
- Include small random perturbations for thorough local exploration

OPTIMIZATION STATE:
- Current iteration: {iteration}
- Is stagnating: {is_stagnating}
- Progress: {progress:.1%}

DESIGN HINT: step_size = base_step * (1 - progress). Direction = normalize(leader - position) + small_random_vector.
""",
    ),
    PromptTemplate(
        name="gradient_estimate",
        description="Gradient-like directed movement toward leader",
        operator_type="local_search",
        template="""Design a PSEUDO-GRADIENT operator that mimics gradient descent toward the leader.

REQUIREMENTS:
- Move in the direction of (leader - position) with controlled step size
- Step size proportional to distance but capped to prevent overshooting
- Add momentum-like behavior using random dampening
- Avoid steps that are too large

OPTIMIZATION STATE:
- Progress: {progress:.1%}
- Distance metrics available in position and leader arrays

DESIGN HINT: direction = leader - position; step = min(learning_rate, 0.1 * np.linalg.norm(direction)) * direction / (norm + epsilon)
""",
    ),
    PromptTemplate(
        name="coordinate_descent",
        description="One dimension at a time optimization",
        operator_type="local_search",
        template="""Design a COORDINATE DESCENT inspired operator.

REQUIREMENTS:
- Focus perturbation on ONE or FEW dimensions at a time
- Select dimensions based on largest difference from leader
- Make a directed step in those dimensions toward leader values
- Leave other dimensions mostly unchanged (small noise only)

OPTIMIZATION STATE:
- Problem dimension: {dim}
- Selected features: {num_selected}

DESIGN HINT: Find top-k dimensions with largest |position - leader| difference, update mainly those.
""",
    ),
]


# Diversification Operators - Detailed English Templates
DIVERSIFICATION_TEMPLATES = [
    PromptTemplate(
        name="explosion_operator",
        description="Population explosion for escaping deep local optima",
        operator_type="diversification",
        template="""Design an EXPLOSION operator to escape local optima through large perturbations.

REQUIREMENTS:
- Create a LARGE jump from current position (not small refinement)
- Direction should be somewhat strategic (not purely random):
  * Consider moving AWAY from leader if stuck (opposition)
  * Or toward unexplored regions
- Magnitude should be proportional to stagnation severity
- Still maintain some structure (not completely random initialization)

OPTIMIZATION STATE:
- Stagnation duration: {stagnation_count} iterations without improvement
- Current diversity: {diversity:.1%}
- Progress: {progress:.1%}

DESIGN HINT: explosion_magnitude = base_magnitude * (1 + stagnation_count / 10). Direction can be random unit vector or anti-leader direction.
""",
    ),
    PromptTemplate(
        name="opposition_based",
        description="Opposition-based learning for diversification",
        operator_type="diversification",
        template="""Design an OPPOSITION-BASED LEARNING operator.

REQUIREMENTS:
- Calculate the opposite point: opposite = lower_bound + upper_bound - position
- For our bounds: opposite = -10 + 10 - position = -position (when bounds are symmetric)
- Use PARTIAL opposition: blend between current and opposite
- Opposition ratio should depend on stagnation severity

FORMULA:
partial_opposite = position + opposition_rate * (opposite - position)
where opposition_rate in [0, 1], higher when more stagnated

OPTIMIZATION STATE:
- Stagnation count: {stagnation_count}
- Current diversity: {diversity:.3f}

DESIGN HINT: opposition_rate = min(1.0, 0.3 + 0.05 * stagnation_count)
""",
    ),
    PromptTemplate(
        name="levy_flight",
        description="Levy flight for heavy-tailed exploration",
        operator_type="diversification",
        template="""Design a LEVY FLIGHT operator for exploration with heavy-tailed step distribution.

REQUIREMENTS:
- Implement Levy flight: mostly small steps, occasionally large jumps
- Levy distribution approximation: step ~ u / |v|^(1/beta), where u,v ~ Normal(0,1)
- Common beta = 1.5 (between Gaussian beta=2 and Cauchy beta=1)
- Direction should be random or semi-guided toward leader
- Scale the step appropriately for the search space

OPTIMIZATION STATE:
- Progress: {progress:.1%}
- Is stagnating: {is_stagnating}

DESIGN HINT: 
sigma = (gamma(1+beta)*sin(pi*beta/2) / (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta)
step = 0.01 * u * sigma / |v|^(1/beta) * (leader - position)
""",
    ),
    PromptTemplate(
        name="cauchy_mutation",
        description="Cauchy distribution mutation for heavy-tailed exploration",
        operator_type="diversification",
        template="""Design a CAUCHY MUTATION operator using Cauchy distribution for exploration.

REQUIREMENTS:
- Use Cauchy distribution (heavy tails) instead of Gaussian
- Cauchy has higher probability of large jumps
- Scale parameter should adapt to stagnation level
- np.random.standard_cauchy() generates Cauchy(0,1) samples

OPTIMIZATION STATE:
- Stagnation count: {stagnation_count}
- Diversity: {diversity:.3f}
- Progress: {progress:.1%}

DESIGN HINT: scale = base_scale * (1 + stagnation_count * 0.1); 
new_position = position + scale * np.random.standard_cauchy(len(position))
""",
    ),
]


# All templates combined
ALL_TEMPLATES = (
    MUTATION_TEMPLATES
    + CROSSOVER_TEMPLATES
    + LOCAL_SEARCH_TEMPLATES
    + DIVERSIFICATION_TEMPLATES
)

TEMPLATE_BY_NAME = {t.name: t for t in ALL_TEMPLATES}
TEMPLATES_BY_TYPE = {
    "mutation": MUTATION_TEMPLATES,
    "crossover": CROSSOVER_TEMPLATES,
    "local_search": LOCAL_SEARCH_TEMPLATES,
    "diversification": DIVERSIFICATION_TEMPLATES,
}


def format_prompt(template: PromptTemplate, state: Dict[str, Any]) -> str:
    """
    Format a prompt template with current optimizer state.

    Args:
        template: The prompt template to use
        state: Current optimizer state dictionary

    Returns:
        Formatted prompt string
    """
    # Default values for missing state keys
    defaults = {
        "iteration": 0,
        "max_iteration": 100,
        "progress": 0.0,
        "diversity": 0.5,
        "improvement": 0.0,
        "mean_cost": 1.0,
        "is_stagnating": False,
        "dim": 30,
        "num_selected": 15,
        "stagnation_count": 0,
    }

    # Merge state with defaults
    context = {**defaults, **state}

    # Format the template
    try:
        formatted = template.template.format(**context)
    except KeyError:
        # If a key is missing, use a simple version
        formatted = template.template

    return formatted


def get_prompt_for_stagnation(state: Dict[str, Any]) -> tuple[str, PromptTemplate]:
    """
    Select appropriate prompt based on stagnation type and optimization phase.

    Args:
        state: Current optimizer state

    Returns:
        Tuple of (formatted_prompt, template_used)
    """
    diversity = state.get("diversity", 0.5)
    progress = state.get("progress", 0.5)
    stagnation_count = state.get("stagnation_count", 0)

    # Decision logic based on optimization state
    if diversity < 0.1 or stagnation_count > 15:
        # Critical: Very low diversity or severe stagnation - need diversification
        templates = DIVERSIFICATION_TEMPLATES
    elif progress < 0.3:
        # Early stage - prefer mutation for exploration
        templates = MUTATION_TEMPLATES
    elif progress > 0.7:
        # Late stage - prefer local search for exploitation
        templates = LOCAL_SEARCH_TEMPLATES
    else:
        # Middle stage - crossover works well for combining good solutions
        templates = CROSSOVER_TEMPLATES

    # Select random template from appropriate category
    import random

    template = random.choice(templates)

    return format_prompt(template, state), template


def build_full_prompt(template: PromptTemplate, state: Dict[str, Any]) -> str:
    """Build complete prompt with system instructions."""
    user_prompt = format_prompt(template, state)
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"
