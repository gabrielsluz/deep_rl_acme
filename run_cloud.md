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
=> Usar JupyterLab para facilitar envio de código e pegar resultados: https://www.youtube.com/watch?v=kyNbYCHFCSw 


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
    - => Disco permanente? Bucket? Como acessa dentro da VM? Talvez tenha que ter dois passos 
        Container linka volume com VM no lugar onde passa para o bucket.
    - Ter checkpoints contínuos, não só no final.
    - Disk ou Bucket? => Disk
- Como iniciar com um container Docker?
- GPU Quota => fazer upgrade para conta paga
- Qual configuração: CPU, RAM, GPU?
- Como acessar com SSH?
    - Afeta pegar os arquivos?

1 Passo:
- Criar git repo com o código leve.
- Criar imagem docker para fazer clone e rodar o código.
- Guardar os resultados em algum lugar que eu possa acessar.
- Interromper a máquina programaticamente.
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


VM:
Imagem: https://hub.docker.com/repository/docker/gabrielsluz1999/acme_rl/general
Como usar a imagem?:
- Modo interativo: docker run -it --rm --gpus all gabrielsluz1999/acme_rl:...
Preciso colocar o código a partir do github usando volume para rodar no container.
Como colocar o código na VM?

Ideia - inicia, roda, mata:
- Entrypoint do container é o script que:
    - Clona o código do github
    - Roda o experimento
    - Salva os resultados no bucket
    - Para a VM
    - Separar código e resultados. => código leve para clone ser rápido.

Ideia - inicia, usa, interrompe, usa ...
=> Custos de manter o estado. => acho melhor não.
