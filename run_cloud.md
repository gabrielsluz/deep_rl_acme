Ideia:
- Tese:
    - Deep NN funcionam. RL, DQN e PPO, funcionam. E pode funcionar nessa tarefa de pushing com orientação.
    - Rodar com GPU vai permitir iterar rapidamente.
    - Modernizar a rede neural e o treinamento deve fazer uma grande diferença.
    - A biblioteca ACME é bem otimizada, com jax, optax e rlax. Vale a pena continuar com ela.
- Passos:
    - Criar a imagem que rode no Docker => crrigir os erros.
    - Definir uma Cloud com GPUs que rode com Docker.
        - GCP => 300 dolares em creditos.
        - Subir uma VM com GPU
        - Rodar um container docker meu.
    - Testar a cloud e fazer um teste de sanidade => 100 epocas...
    - Medir o tempo de experimentação e ver se vale a pena.
    - Passar a limpo o treinamento, parâmetros e premissas.
        - Retirar rendering da simulação => if render: ...
        - Parar antes do max_steps se ficar preso => problema: robô nunca aprende que isso é ruim.
            - Introduzir recompensa negativa por murrinhar?
    - Acaompnhar o treinamento da rede neural => loss de treino e de validação?



Como rodar uma VM no GCP?
https://www.youtube.com/watch?v=jh0fPT-AWwM
- Opções:
    - Kubernetes
    - Cloud Run
    - Google Compute Engine => mais simples.
        - VMs
        - Container Optmized OS
- Guardar e manage docker images: Google Container Registry (GCR)
- Como minimizar o custo?

Tópicos:
- Como criar uma VM, iniciar, parar?
- Como colocar e pegar arquivos da VM?
- Como iniciar com um container Docker?
- GPU Quota => fazer upgrade para conta paga
- Qual configuração: CPU, RAM, GPU?
- Como acessar com SSH?
    - Afeta pegar os arquivos?

1 Passo:
- Rodar uma VM sem GPU com container.
- Aprender a transferir arquivos par algum lugar que eu possa acessar e fazer download.
- Parar a máquina manualmente.
- Aprender a parar automaticamente.

Teste full:
- Iniciar VM com GPU
- Checar se está usando GPU
- Rodar experimento
- Experimento rodando bem rápido que na minha máquina.
- Pegar resultados e gravar na minha máquina => logs + checkpoint modelo.
