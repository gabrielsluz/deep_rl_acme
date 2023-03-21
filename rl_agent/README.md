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
- A política está sendo atualizada no actor? sim
- O treinamento está sendo feito? sim
- Observações estão sendo concatenadas por n_steps? nao
- O que é td-learning por n_steps? 

Experimentos:
- Definir quais os parametros importantes
- Definir como rodar experimentos => gambiarra por enquanto.
    - Guardar: logs, parametros
- Fazer uma busca de hiperparametros considerando-os independentes.
- Rodar para objeto retangular


Parametros importantes:
- Experimento:
    - n_epochs
    - ep_per_epoch: 20
- Env:
    - FrameStackDepth: [1,2,4]
    - max_steps: 100
    - reward: projection, outra que seja mais adequada a objetos de diferentes formas
- Otimizador:
    - learning_rate: 1e-3
    - batch_size: 2048 => so comeca a treinar a partir da epoca 3.
- RL:
    - Epsilon decay
    - discount: 0.95
- Learner:
    - target_update_period
    - observations_per_step
    - min_replay_size = batch_size

Próximos passos:
- Experimento com 300 epocas => eps decay de 150 epocas
- Se resultado ficar ruim: Avaliar desempenho por objeto e gravar video
    - Checkpoint + video
- Funcao de recompensa
- Orientacao
- Estudar formation control
