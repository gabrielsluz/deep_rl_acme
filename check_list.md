# Garantir que está funcionando
Checklist:
- Função de recompensa:
    - Depois de empurrar na direção correta e obter recompensa positiva, ao não empurrar o objeto obtém
    recompensa igual. O mesmo ocorre com recompensas negativas.
        - Explicação: o objeto continua se movendo, bem pouquinho, na direção.
        - Solução: usar a magnitude do movimento também. => Piorou o resultado.
    - Recompensa de sucesso é pequena: 1.00. Enquanto acertar a direção é 0.33. E penalidade de tempo dá -0.09.
        - Problema talvez quando o agente encosta no objeto na direção errada => toma grande recompensa negativa.
    - Penalidade de morte tem que ser alta => senão vale mais a pena se matar que ganhar uma série de recomepensas
        negativas. => ou deixar  a penalidade de tempo baixa.
    - Testar no playground como ela se comporta e se ele direciona para o melhor caminho.
        - Reward shaping
    - Resultados:
        - Alta punição para morte e recompensa para sucesso, não melhoraram os resultados. 
        - Projection reward foi melhor que a progress
        - Avaliar se a rede convolucional enxerga o limite de exploração. Avaliar se dá para retirar a morte súbita.
    - E se não tivessemos recompensa ou punição para morte ou sucesso? => talvez ele comece a se matar e a andar em circulos.
- Observações:
    - Garantir no playground que estão corretas.
    - Faz sentido incluir a distância do objetivo, medida de acordo com o raio do robô?
        - Talvez no próprio primeiro canal. Ou adicionando um canal novo
    - FrameStack => Ajuda ou atrapalha? Daria para fazer um a longo prazo?
    - A rede convolucional consegue identificar o limite de morte?
        - Introduzir mais um canal com distância até o objeto?
    - Introduzir dois canais: dist to obj, dist to goal, piorou em muito o resultado. Hipótese: aumentou demais a entrada
        e o tamanho da rede. Possível solução: rede com 2 pathways: faz embedding da imagem e concatena com info de posição.
- Treinamento:
    - Epsilon com decaimento linear e depois um longo tempo exploitando foi a melhor exploração encontrada.
    - Batch size maior ajuda a estabilizar o treino.
    - target_update_period = 100 => parece alto, dado ao problema de ficar procurando um moving target.
        - Usando 2000 atingiu um resultado de 73% de sucesso, comparado com 54% com 100.
        - Vale a pena testar um maior. 4000. Atingiu 71%. 
        - Fazer testes usando parâmetros de artigos, inclusive os do Alysson
            - Learning rate: 5e-4 => 64% secesso.
        - Testar com gamma = 0.99 => 71% sucesso
    - Multi step Q Learning => e-greedy atrapalha => tem que ser on policy. Vou testar epsilon de 1%: 68% sucesso.
        - Pode ser mais útil quando tivermos muitas épocas de exploitation.
    - "Curriculum learning" => épocas iniciais com safe zone maior, ir apertando a gradualmente até chegar na precisão alvo.
        - Fazer os primeiros experimentos sempre com uma safe zone maior para que a tarefa seja factível. 
    - 100 epocas para safe zone = 1, dá 64% de sucesso
    - 300 épocas: 91% (com max_Steps = 400 e eps= 0.05), 85% no jeito tradicional.
    - n-step => fazer um teste com n = 3. => piorou: 58% comparado com 64%
    - Experimentar com os parâmetros do replay
- Avaliação:
    - Usar epsilon?
        - Com eps = 0 => 85%. Com eps = 0.005 => 92% Com eps = 0.05 => 87%
    - Raio de morte do objeto => problemático, pois altera o input do agente => gerou perda de desempenho
    - max_steps
    
- Rede neural:
    - Funcionou bem com uma rede pequena, mas vale a pena tentar melhorar, pois é a alma do algoritmo.
        - Parâmetros afetados por tornar a rede neural maior: 
            - learning_rate, update_period, batch_size. E talvez outros em segunda ordem.
    - LeNet-5 : Melhorou resultado em 100 epocas de 64% para 82%
- Agente:
    - Para conseguir um controle bem fino, talvez seja necessário usar ações contínuas.

Melhorias:
- Buscar melhorias algoritmicas para lidar com recompensas esparsas e de longo prazo.
- Simplificar função de recompensa e condição de morte?