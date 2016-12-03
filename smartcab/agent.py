import random
import math
import pandas as pd
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, desvio = 0, gamma = 0, ajuste = 0.95, optimized = False):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
	#Q-table usado: DataFrame Qdf
	self.Qdf = pd.DataFrame({'light':[],
				 'oncoming': [],
				 'waypoint': [],
				 'Qvalue': [],
				 'action': []})

       
	self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
	
        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
	self.trial = 1.0 # numero da tentativa. Sera usado para diminuir o epsilon (exploration factor)
	self.acoes = pd.DataFrame({'light':[],
				 'oncoming': [],
	  		         'left': [],
				 'right': [],
				 'waypoint': [],
				 'Qvalue': [],
				 'action': [],
				 'trial': [],
				 'testing': [],
				 'reward':[]})  

	self.desvio = desvio #valor inicial dos Q-valores
	self.gamma = gamma
	self.ajuste = ajuste
	self.optimized = optimized
	self.testing = None
	
    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
	self.testing = testing
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
	

	if self.optimized:
	    self.epsilon = self.ajuste ** self.trial
	else:
	    self.epsilon = self.epsilon - 0.05
	
	self.trial = self.trial + 1

    	self.novoEstado = pd.DataFrame()
	self.estadoPassao = pd.DataFrame()
	# Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
	if testing == True:
		self.alpha = 0
		self.epsilon = 0
	


        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline
	
        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0
        self.estadoPassado = self.novoEstado
	state = {'waypoint': waypoint,
		 'inputs': inputs,
		 'deadline': deadline}	
	


	for i, value in state['inputs'].iteritems():
	    if value == None:
		state['inputs'][i] = 'parado'
		
	self.novoEstado = pd.DataFrame({'light':[state.get('inputs').get('light')],
			      'oncoming': [state.get('inputs').get('oncoming')],
			      'waypoint': [state.get('waypoint')],
			      'Qvalue': 0})

	return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

	check = pd.DataFrame({
		'light': pd.Series(self.Qdf['light'] == state['light'][0]) |
			 pd.Series(self.Qdf['light'].isnull() & state['light'].isnull()),
		'waypoint': pd.Series(self.Qdf['waypoint'] == state['waypoint'][0]) |
			 pd.Series(self.Qdf['waypoint'].isnull() & state['waypoint'].isnull()),
		'oncoming': pd.Series(self.Qdf['oncoming'] == state['oncoming'][0]) |
			    pd.Series(self.Qdf['oncoming'].isnull() & state['oncoming'].isnull())
		}).all(axis = 1)

	selecao = self.Qdf[check] #Seleciona apenas as acoes correspondentes ao estado

	
	#Encontra a acao com o maximo de Q-value na selecao
	indices = selecao[selecao["Qvalue"] == selecao["Qvalue"].max()].index.values[0]
	action = selecao.loc[indices,'action']


	if action == 'parado':
	    action = None

	#maximo Q-value na selecao
	maxQ = selecao.loc[indices,'Qvalue']

        return {'acao': action, 'maxQ': maxQ} 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """
	
	
	#Cria um DataFrame com o novo estado
	luz = state.get('inputs').get('light')
	oncoming = state.get('inputs').get('oncoming')
	waypoint = state.get('waypoint')
	
	desvio = self.desvio
	d = {'light':[luz,luz,luz,luz],
	     'oncoming': [oncoming, oncoming,oncoming,oncoming],
	     'waypoint': [waypoint,waypoint,waypoint,waypoint],
	     'Qvalue': [np.random.random_sample()/100 + desvio,np.random.random_sample()/100 + desvio,
			np.random.random_sample()/100 + desvio,np.random.random_sample()/100 + desvio],
	     'action': ['parado','left','right','forward']}
	

	novoEstado = pd.DataFrame(d)

	########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
	
	#Checa se a linha ja aparece e
	#Insere o novoEstado na Q table com o Q-value igual a 0, se ele nao estiver na Q table 

	if self.Qdf.empty:
            self.Qdf = self.Qdf.merge(novoEstado,how='outer')
	
	else:
	    check = pd.DataFrame({
		'light': pd.Series(self.Qdf['light'] == novoEstado['light'][0]) |
			 pd.Series(self.Qdf['light'].isnull() & novoEstado['light'].isnull()),
		'waypoint': pd.Series(self.Qdf['waypoint'] == novoEstado['waypoint'][0]) |
			 pd.Series(self.Qdf['waypoint'].isnull() & novoEstado['waypoint'].isnull()),
		'oncoming': pd.Series(self.Qdf['oncoming'] == novoEstado['oncoming'][0]) |
			    pd.Series(self.Qdf['oncoming'].isnull() & novoEstado['oncoming'].isnull())
		})

	    if not check.all(axis = 1).any():
		self.Qdf = self.Qdf.merge(novoEstado,how='outer')

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()


        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action        	

	aleatorio = np.random.random_sample()

	if self.learning == False:
	    action = self.valid_actions[random.randrange(0,3)]
	else:
	    if aleatorio < self.epsilon:
	    	action = self.valid_actions[random.randrange(0,3)]
	    else:
		temp = self.get_maxQ(self.novoEstado)
		action = temp['acao']



        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
 	

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########

	novoEstado = self.novoEstado

	if action == None:
	    action = 'parado'
	# When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

	#Posicao do Qmax
	check = pd.DataFrame({
		'light': pd.Series(self.Qdf['light'] == novoEstado['light'][0]) |
			 pd.Series(self.Qdf['light'].isnull() & novoEstado['light'].isnull()),
		'waypoint': pd.Series(self.Qdf['waypoint'] == novoEstado['waypoint'][0]) |
			 pd.Series(self.Qdf['waypoint'].isnull() & novoEstado['waypoint'].isnull()),
		'oncoming': pd.Series(self.Qdf['oncoming'] == novoEstado['oncoming'][0]) |
			    pd.Series(self.Qdf['oncoming'].isnull() & novoEstado['oncoming'].isnull()),
		'action': pd.Series(self.Qdf['action'] == action)
		})
	

	Qmax = self.Qdf.iloc[check[check.all(axis = 1)].index[0]]['Qvalue']

	if self.estadoPassado.empty:
	    QmaxFuturo = self.desvio
	else:
            QmaxFuturo = self.get_maxQ(self.estadoPassado)['maxQ']

	#Atualiza o valor Q, aprendendo reward e o Q-value para o estado futuro
	self.Qdf.loc[check[check.all(axis = 1)].index[0],'Qvalue'] = (1 - self.alpha) * Qmax + self.alpha * (reward + self.gamma * QmaxFuturo)

	temp = self.Qdf.loc[check.all(axis = 1)].reset_index()

	temp.loc[0,'reward'] = reward

	temp.loc[0,'trial'] = self.trial - 1

	temp.loc[0,'testing'] = self.testing

	self.acoes = self.acoes.append(temp, ignore_index = True)

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action

        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run(optimized = False, epsilon = 1, alpha = 0.5, desvio = 0, gamma = 0, ajuste = 0, n_test = 10, learning = False):
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
 
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = False)
 


    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1 [1, 0.9, 0.75]
    #    * alpha   - continuous value for the learning rate, default is 0.5 [0, 0.5, 1]
    #    * desvio - constante que aumenta o Q inicial para forcar o agente a explorar novas opcoes, se os Q iniciais sao inicializados como zero o agente fica mais propenso a permanecer em caminhos sub otimos [0, 0.5, 1, 2,4]
    #    * gamma - valor de desconto do Q otimo futuro. [0, 0.5]
    #    * a - constante que entra na funcao de decaimento do epsilon. Quanto maior, mais rapido o epsilon cai e menos o agente explora os estados. [0.95, 0.9]
    #    * optimized - define se a funcao de decaimento do epsilon a ser usada e a otimizada ou a default
    agent = env.create_agent(LearningAgent, learning = learning, epsilon = epsilon, alpha = alpha, desvio = desvio, gamma = gamma, ajuste = ajuste, optimized = optimized)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.0001, log_metrics = True, display = False, optimized = optimized,
		    epsilon = epsilon, alpha = alpha, desvio = desvio, gamma = gamma, ajuste = ajuste)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = n_test, tolerance = 0.05)
    if sim.optimized:
	NomeArquivo = "_epsilon" + str(epsilon) + "_alpha" + str(alpha) + "_desv" + str(desvio) + "_gamma" + str(gamma) + "_ajuste" + str(ajuste) 
	agent.Qdf.to_csv('analises/QdfOtimizado'+NomeArquivo+'.csv')
    	agent.acoes.to_csv('analises/acoesOtimizado'+NomeArquivo+'.csv')
    else:
	if learning == True:
	    agent.Qdf.to_csv('analises/Qdf.csv')
            agent.acoes.to_csv('analises/acoes.csv')

if __name__ == '__main__':
#	run(optimized = False, n_test = 10, learning = False) 
	run(optimized = False, n_test = 10, learning = True)

#	desvio = [0, 0.5, 1, 2]
#	gamma = [0, 0.5]
#	alpha = [0, 0.5, 1]
#	epsilon = [1, 0.75]
#	ajuste = [0.95, 0.9]

#	for i in desvio:
#	    for j in gamma:
#		for z in alpha:
#		    for e in epsilon:
#		        for a in ajuste:
#		            run(optimized = True, epsilon = e, alpha = z, desvio = i, gamma = j, ajuste = a, n_test = 50, learning = True)

   	




