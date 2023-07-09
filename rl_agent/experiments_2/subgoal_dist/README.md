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
- Tamanho do caminho
- Objeto
- Número de subgoals.


Passagem de parâmetros:
- dataclass de config
- Arquivos a modificar:
    - box2D_img_pushing_pose_env.py
    - PushSimulatorPose.py
    - pose_subgoal_env.py
