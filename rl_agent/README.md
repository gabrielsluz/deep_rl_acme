# RL Agent for testing ACME installation

Environment: CartPole - gym
Agent: DQN


Environment: Gym com o GymWrapper.
Agente: 
- Sonnet + TF para fazer a rede neural? Ou Jax?
- from acme.agents.jax import dqn
    - Define o Config e o init => network

Logging, snapshoting and checkpointing

E-greedy:
- Behavior policy recebe epsilon como argumento.
    - Como modificar o epsilon?
    - Como a behavior_policy é chamada na hora de agir?

Checar:
- Epsilon-greedy
- Discount factor
- Update rate -  RateLimiter + conditionals in code
- Fazer meu próprio Logger
- Rede neural linear
- Usar factories para criar o experimento, do jeito recomendado no artigo
- Como avaliar a política?


Checar na implementacao:
- Environment loop está rodando corretamente? sim
- Discount factor está sendo aplicado o do parâmetro? sim
- Epsilon está decaindo e sendo usado pelo rlax? sim
- A política está sendo atualizada no actor?
- O treinamento está sendo feito?
- Observações estão sendo contenadas por n_steps?
- O que é td-learning por n_steps?