import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import acme
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

import dqn

from acme.wrappers import GymWrapper
# from wrappers.add_channel_wrapper import AddChannelDimWrapper
from wrappers.frame_stack_wrapper import FrameStackWrapper
from environment_loop import EnvironmentLoop
from research_envs.envs.box2D_img_pushing_env import Box2DPushingEnv
from research_envs.envs.rewards import RewardFunctions
from observers.success_observer import SuccessObserver
from acme.utils.loggers import InMemoryLogger

# Utils
def calc_suc_rate(data: list) -> float:
    suc_cnt = 0
    for i in data:
        suc_cnt += i['success']
    return suc_cnt / len(data)

# ENV
def create_environment():
    env = Box2DPushingEnv(smoothDraw=False, reward=RewardFunctions.PROJECTION, max_steps=200)
    env = GymWrapper(env)
    env = FrameStackWrapper(env, frameStackDepth=4)
    return env

def main():
    jax.config.update('jax_enable_x64', True)

    env = create_environment()
    env_spec = specs.make_environment_spec(env)
    # print(env_spec)
    
    # Calculate how big the last layer should be based on total # of actions.
    action_spec = env_spec.actions
    action_size = np.prod(action_spec.shape, dtype=int)
    # print(action_spec, action_size)

    # AGENT
    def network_fn(obs):
        network = hk.Sequential([
            hk.Conv2D(output_channels=8, kernel_shape=[4, 4], stride=4, padding='valid'),
            jax.nn.relu,
            hk.Conv2D(output_channels=16, kernel_shape=[3, 3], padding='valid'),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(env_spec.actions.num_values)
        ])
        x = obs
        x = network(x)
        return x

    dummy_action = utils.zeros_like(env_spec.actions)
    dummy_obs = utils.add_batch_dim(utils.zeros_like(env_spec.observations))

    mlp = hk.without_apply_rng(hk.transform(network_fn))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: mlp.init(rng, dummy_obs),
        apply=mlp.apply
    )

    agent = dqn.DQN(
        environment_spec=env_spec, 
        network=network, 
        batch_size=2048,
        # prefetch_size=4,
        # target_update_period=100,
        observations_per_step=50.0,
        min_replay_size=2048,
        # max_replay_size=1000000,
        # importance_sampling_exponent=0.2,
        # priority_exponent=0.6,
        # n_step=4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=20*130,
        learning_rate=1e-3,
        discount=0.95,
        # seed=1,
    )
    observers = [
        SuccessObserver()
    ]
    logger = InMemoryLogger()

    loop = EnvironmentLoop(env, agent, logger=logger, observers=observers)
    ep_per_epoch = 20
    for epoch_i in range(250):
        loop.run(num_episodes=ep_per_epoch)
        suc_rate = calc_suc_rate(loop._logger.data[-ep_per_epoch:])
        print('Epoch {}: Success Rate: {:.3f}'.format(epoch_i, suc_rate))

    """
    Faltando:
        Concatenar frames => FrameStack => Wrapper do Env. Talvez fazer um wrapper no Gym e usar o
            GymWrapper.
        Checar:
            - Rede Neural funciona, recebe entradas na dimensao correta, passa o batch certinho?
            - Tamanho do batch
            - Steps de learner por steps de actor => 1 learner por episodio => 1 por 20-100 observations?
            - Atualizar a target a cada 5 epocas
    Objetivo:
        - Encontrar os parâmetros que atingem o melhor treinamento consistentemente.
            - Definir quais são os parâmetros chave e
            - Coletar os dados e colocar em csv => plotar os gráficos 
            - Guardar os parâmetros
        - Usar esses parâmetros para retângulo        

    Checkpoint do treinamento para retangulo 95.6% success: /tmp/tmpzb9ufx3z
    Triangulo 91,6% success: /tmp/tmpnxpjpi0h

    Algoritmos candidatos:
        - PPO (Primeiro a tentar)
        - R2D2
        - IMPALA
    Focar no PPO e no Rainbow. Entender onde eles se encaixam no cenário de RL.

    Modificações no Env:
        - Receber objetos por parâmetro e aleatorizar.
        - Checar se o truncation está sendo feito corretamente.
        - Salvar videos de episódios de eval amostrados aleatoriamente.
        - Definir manualamente objetos e area de segurança
        - EVitar de criar episódios que terminam com sucesso em um step
    Modificações no Loop:
        - Adequar ao modo de avaliação: 
            - Ter epocas de eval de 5 - 20 episódios
                - Qual valor de epsilon? As vezes é util para sair de loops.
            - Rodar múltiplas vezes com diferentes random seeds
            - Avaliar o desempenho nos últimos x epsiódios
        - Gravar os resultados: 
            - Treino: reward médio e success rate
            - Eval: reward médio e success rate
        - Log em uma pasta fácil de achar
        - Checkpoints e snapshots
        - Ter um script para rodar experimentos com varias random seeds.
        - Gerar aqueles gráficos bonitos 

    """

if __name__ == '__main__':
    main()
    print("End of code")