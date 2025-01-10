# Cronograma de Estudos para Machine Learning (8 Semanas)  

Este repositório contém um plano detalhado de estudos para aprender Machine Learning em 8 semanas, com foco em teoria, prática e exercícios.  

## Sumário  
- [Semana 1: Introdução ao Machine Learning](#semana-1-introdução-ao-machine-learning)  
- [Semana 2: Preparação de Dados](#semana-2-preparação-de-dados)  
- [Semana 3: Algoritmos de Machine Learning - Supervisão](#semana-3-algoritmos-de-machine-learning---supervisão)  
- [Semana 4: Algoritmos de Machine Learning - Não Supervisão](#semana-4-algoritmos-de-machine-learning---não-supervisão)  
- [Semana 5: Redes Neurais e Deep Learning](#semana-5-redes-neurais-e-deep-learning)  
- [Semana 6: Aprendizado por Reforço](#semana-6-aprendizado-por-reforço)  
- [Semana 7: Avaliação e Otimização](#semana-7-avaliação-e-otimização)  
- [Semana 8: Projeto Final e Próximos Passos](#semana-8-projeto-final-e-próximos-passos)  

---

## Semana 1: Introdução ao Machine Learning  

### **Teoria (1h)**  
- O que é Machine Learning e sua importância em diferentes áreas.  
- Tipos de aprendizado: supervisionado, não supervisionado e por reforço.  
- Componentes principais: dados, modelos, treinamento, validação e teste.  
- Ferramentas e bibliotecas populares: Python, NumPy, pandas, scikit-learn, TensorFlow, PyTorch.  

### **Prática (1h)**  
- Instalar ferramentas básicas: Python, Jupyter Notebook, e bibliotecas como NumPy, pandas e scikit-learn.  
- Explorar um conjunto de dados simples (ex: Titanic ou Iris) com pandas.  
- Criar gráficos básicos com Matplotlib ou Seaborn.  

### **Exercício**  
- Analisar estatísticas descritivas do dataset Iris usando pandas e criar gráficos de dispersão para entender relações entre variáveis.  

---

## Semana 2: Preparação de Dados  

### **Teoria (1h)**  
- A importância da qualidade dos dados.  
- Processos de pré-processamento: limpeza, normalização, transformação e redução de dimensionalidade.  
- Divisão de dados em conjuntos de treino, validação e teste.  

### **Prática (1h)**  
- Limpar dados com pandas: lidar com valores ausentes e duplicados.  
- Escalar e normalizar dados com scikit-learn.  
- Dividir um conjunto de dados em treino/teste usando `train_test_split`.  

### **Exercício**  
- Carregar o dataset Titanic, tratar valores ausentes e normalizar variáveis numéricas.  

---

## Semana 3: Algoritmos de Machine Learning - Supervisão  

### **Teoria (1h)**  
- Algoritmos supervisionados comuns: regressão linear, regressão logística, árvores de decisão, e SVM.  
- Métricas de avaliação: accuracy, precision, recall, F1-score, RMSE.  

### **Prática (1h)**  
- Treinar um modelo de regressão linear em dados simples (ex: preço de casas).  
- Treinar um modelo de classificação com regressão logística no dataset Iris.  
- Avaliar modelos usando `cross_val_score` e relatórios de classificação.  

### **Exercício**  
- Criar um modelo de árvore de decisão para prever a sobrevivência no Titanic, avaliar e interpretar os resultados.  

---

## Semana 4: Algoritmos de Machine Learning - Não Supervisão  

### **Teoria (1h)**  
- Introdução ao aprendizado não supervisionado: clustering e redução de dimensionalidade.  
- Algoritmos comuns: K-Means, DBSCAN, PCA.  

### **Prática (1h)**  
- Aplicar K-Means para agrupar dados do dataset Iris.  
- Reduzir dimensionalidade com PCA e visualizar os resultados.  

### **Exercício**  
- Implementar K-Means em um conjunto de dados desconhecido e analisar os clusters formados.  

---

## Semana 5: Redes Neurais e Deep Learning  

### **Teoria (1h)**  
- O que são redes neurais artificiais (ANNs).  
- Estrutura básica: neurônios, camadas, funções de ativação e perda.  
- Introdução ao TensorFlow e Keras.  

### **Prática (1h)**  
- Construir e treinar uma rede neural simples em TensorFlow/Keras para classificação de dígitos (MNIST).  
- Avaliar o modelo e ajustar hiperparâmetros básicos.  

### **Exercício**  
- Construir uma rede neural para prever se uma pessoa sobreviveu no Titanic.  

---

## Semana 6: Aprendizado por Reforço  

### **Teoria (1h)**  
- Introdução ao aprendizado por reforço: agentes, estados, ações e recompensas.  
- Conceitos básicos: Q-Learning, redes neurais profundas no aprendizado por reforço (DQN).  

### **Prática (1h)**  
- Implementar um agente simples com Q-Learning para resolver o problema do "Frozen Lake" no OpenAI Gym.  
- Explorar o impacto de parâmetros como taxa de aprendizado e fator de desconto.  

### **Exercício**  
- Treinar um agente que jogue "CartPole" usando o OpenAI Gym.  

---

## Semana 7: Avaliação e Otimização  

### **Teoria (1h)**  
- Técnicas de avaliação: validação cruzada, curva ROC, AUC.  
- Otimização de hiperparâmetros com Grid Search e Random Search.  
- Overfitting e técnicas de regularização.  

### **Prática (1h)**  
- Avaliar modelos com curvas ROC e AUC no dataset Titanic.  
- Usar GridSearchCV para encontrar os melhores hiperparâmetros de um modelo.  

### **Exercício**  
- Ajustar os hiperparâmetros de uma árvore de decisão para maximizar o desempenho.  

---

## Semana 8: Projeto Final e Próximos Passos  

### **Teoria (1h)**  
- Boas práticas no desenvolvimento de projetos de ML.  
- Introdução a MLOps: deploy e monitoramento de modelos.  
- Ferramentas para seguir aprendendo: Kaggle, Papers with Code, e cursos avançados.  

### **Prática (2h)**  
- Criar um projeto completo:  
  1. Escolher um dataset (ex: Kaggle).  
  2. Analisar, limpar e processar os dados.  
  3. Treinar e avaliar um modelo supervisionado ou não supervisionado.  
  4. Documentar resultados e apresentar insights.  

### **Exercício**  
- Publicar o projeto no GitHub e receber feedback da comunidade.  

---

## Dicas Gerais  
1. **Prática Constante:** Reserve um tempo diário ou semanal para praticar.  
2. **Participação em Comunidades:** Envolva-se em fóruns como Stack Overflow, Medium, e Kaggle.  
3. **Estudo Contínuo:** Explore tópicos avançados, como NLP, Visão Computacional e séries temporais.  

Pronto para começar? 🚀  

