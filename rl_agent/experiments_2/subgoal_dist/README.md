Simplifying:
- Implement subgoal tolerance.
- Fix the initial distance: pos = 120, ori = np.pi
- Vary the subgoal tolerance X max_step pos and orig
- Goal: find the most efficient subgoal tactic and the tradeoff between subgoal tolerance and efficiency.
Tactics:
- 1 shot: sub_tol = final_tol, step: (130, 2*np.pi)
- 2 shots: sub_tol = (2, np.pi), step: (60, np.pi)
- 4 shots: sub_tol = (2, np.pi), step: (30, np.pi)
- 8 shots: sub_tol = (2, np.pi), step: (15, np.pi)
- 16 shots: sub_tol = (2, np.pi), step: (7.5, np.pi)
- 32 shots: sub_tol = (2, np.pi), step: (3.75, np.pi)
- 64 shots: sub_tol = (2, np.pi), step: (3, np.pi)
200 episodes for each tactic.
Later, evaluate with sub_tol = (3, np.pi)

Results are showing that more subgoals decreases a lot performance.

Alternatives:
- Train a position only policy to go to the subgoal. Then, train a pose policy to go to the final goal.
    - To handle obstacles, we train it with an imaginary corridor.
- Improve the handling of getting stuck    







Fazer o experimento para escrever no artigo. Coletar os dados csv e guardar.
Experimento:
easy_pose_6 com os três objetos. Separar os resultados por objeto.
Distâncias iniciais: ori rand(0, 360), pos rand(0, 120)

Treinamento foi feito com: max_dist = 30 e max_ori = np.pi/2

Distâncias de interpolação: d*[1.25, 1, 0.75, 0.5, 0.25]
- pos: 37.5, 30, 22.5, 15.0, 7.5
- ori: 1.9634954084936207, 1.5707963267948966, 1.1780972450961724, 0.7853981633974483, 0.39269908169872414
Experimentos: Fixar pos = 30, variar ori. Fixar ori = 90, variar pos.
Ou usar proporcionais: da distância máxima: 125, 100, 75, 50, 25
Dist_max: 30
Ori_max: 90 = np.pi/2


Tolerância de subgoals e de objetivo iguais:
self.safe_zone_radius = 2
self.orientation_eps = np.pi/36

Episódios: 600

Medir:
- Sucesso
- Tamanho do caminho: steps
- Objeto
- Número de subgoals.
- Distância inicial : ori e pos
- Max dist_step e max ori_step
- real dist_step, real ori_step

max_steps: 1000


Passagem de parâmetros:
- dataclass de config
- Arquivos a modificar:
    - box2D_img_pushing_pose_env.py
    - PushSimulatorPose.py
    - pose_subgoal_env.py


Eu quero facilitar o treinamento diminuindo a distância inicial.
Mas, quanto menor a distância, menos eficiente fica o caminho da poítica.
Então quero encontrar a menor distância que não degrada muito a eficiência.
Além disso, a distância é composta de dois fatores: orientação e posição.

Porém, pela inicialização inicial, pode ocorrer de d = 5 e ori = 90 graus.
Ou seja, pequena distância para posição e grande distância para orientação.
O que eu quero é medir a taxa de sucesso e steps para combinações de d e ori.
Mas, não pego a eficiência do caminho, então não pode ser isso.
Quero ter distância inicial fixa e variar os subgoals.
Deixar a aleatoriedade resolver - 600 episódios. Usar a heurística de andar na direção do obj.
Variar a relação entre d e ori? Ou manter a razão constante?

