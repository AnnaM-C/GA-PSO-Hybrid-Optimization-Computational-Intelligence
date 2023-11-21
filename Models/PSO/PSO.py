
import numpy as np
import torch
import torch.nn as nn
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import operator
import random
import math

class PSO():
    def __init__(self, objective, populationSize, model, device,
                 vmaxInit: float =2.0 ,
                  vminInit : float=0.2,
                  c1 : float =2,
                  c2 : float =2,
                  wmax : float =0.9,
                  wmin : float =0.4,
                  posMinInit : float =-.1,
                  posMaxInit : float =+.1, 
                  pso_type : str ='global',
                  num_neighbours : int = None,
                  interval: float=10,):
        
        self.populationSize = populationSize
        self.dimension=850
        self.posMinInit=posMinInit
        self.posMaxInit=posMaxInit
        self.VMaxInit=vmaxInit
        self.VMinInit=vminInit
        self.interval=interval
        self.wmax=wmax
        self.wmin=wmin
        self.c1=c1
        self.c2=c2

#Parameter setup

# particle rerpresented by list of 5 things
# 1. fitness of the particle,
# 2. speed of the particle which is also going to be a list,
# 3.4. limit of the speed value,
# 5. best state the particle has been in so far.

    def generate(size, smin, smax):
        part = creator.Particle(random.uniform(posMinInit, posMaxInit) for _ in range(size))
        part.speed = [random.uniform(VMinInit, VMaxInit) for _ in range(size)]
        part.smin = smin #speed clamping values
        part.smax = smax
        return part

    def updateParticle(part, best, weight):
        #implementing speed = 0.7*(weight*speed + c1*r1*(localBestPos-currentPos) + c2*r2*(globalBestPos-currentPos))
        #Note that part and part.speed are both lists of size dimension
        #hence all multiplies need to apply across lists, so using e.g. map(operator.mul, ...

        r1 = (random.uniform(0, 1) for _ in range(len(part)))
        r2 = (random.uniform(0, 1) for _ in range(len(part)))

        v_r0 = [weight*x for x in part.speed]
        v_r1 = [c1*x for x in map(operator.mul, r1, map(operator.sub, part.best, part))] # local best
        v_r2 = [c2*x for x in map(operator.mul, r2, map(operator.sub, best, part))] # global best

        part.speed = [0.7*x for x in map(operator.add, v_r0, map(operator.add, v_r1, v_r2))]

        # update position with speed
        part[:] = list(map(operator.add, part, part.speed))

    def assign_weights(self, particle, finalLayer):
        # First step to get the particles weights out from it, convert to an numpy array
        particleweightsNP1 = np.array(particle)

        particleweightsNP = particleweightsNP1[:840]
        biases = np.array(particleweightsNP1[-10:])

        # Putting biases straight in
        biases = torch.from_numpy(biases).float()
        finalLayer.bias = torch.nn.Parameter(biases.float())

        #print(particleweightsNP)

        print("Shape of particle: " + str(particleweightsNP.shape))

        # Converting to the correct shape!
        reshapedWeights = particleweightsNP.reshape(10,84)
        print("Shape of reshaped particle: " + str(reshapedWeights.shape))

        # Convert to torch array!
        torchWeights = torch.from_numpy(reshapedWeights).float()

        # Now we want to set the weights of the finalLayer to these weights

        finalLayer.weight = torch.nn.Parameter(torchWeights.float())


    def evaluate(self, particle, x, y, PopModel):

        PopModel.train()
        prediction=PopModel(x)
        loss=nn.CrossEntropyLoss()(prediction, y)

        return loss,
    

    toolbox = base.Toolbox()
    toolbox.population(n=populationSize) # Population Size


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    toolbox.register("evaluate2", evaluate)
    toolbox.register("particle", generate, size=dimension, smin=-3, smax=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle)

    def search(self, x,y,finalLayer,PopModel):

        for particle in self.pop:

            w = wmax - (wmax-wmin)*g/iterations #decaying inertia weight

            for part in pop:

                # We need to create a new evaluate function that takes a particles values, reshapes, put into final layer, calculates the loss and returns the loss

                #part.fitness.values = toolbox.evaluate(part) #actually only one fitness value
                assign_weights(particle, finalLayer)

                part.fitness.values = toolbox.evaluate2(particle,x,y,PopModel)


                #update local best
                if (not part.best) or (part.best.fitness < part.fitness):   #lower fitness is better (minimising)
                #   best is None   or  current value is better              #< is overloaded
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values

                #update global best
                if (not best) or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

            for part in pop:
                toolbox.update(part, best,w)

            # Gather all the fitnesses in one list and print the stats
            # print every interval
            if g%interval==0: # interval
                logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
                print(logbook.stream)
                #print('best ',best, best.fitness)

        print('best particle position is ',best)

        return best.fitness.values[0]

            