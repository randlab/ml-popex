#!/bin/env python

from popex.popex_objects import Problem, CatParam
from popex import algorithm
from fluvial import FluvialSimulation
from tracer import TracerTest

fluvial_simulation = FluvialSimulation()
tracer_test = TracerTest(steps_factor=4, working_dir='popex-modflow-500', likelihood='sample', modpath='steady-state')

def main():
    problem = Problem(generate_m=fluvial_simulation.generate_m, # model generation function
                      compute_log_p_lik=tracer_test.compute_log_p_lik, # log likelihood (forward)
                      get_hd_pri=fluvial_simulation.get_hd_pri) # prior hard conditioning

    algorithm.run_popex_mp(pb=problem,
                           path_res='popex-500/',
                           path_q_cat='./',
                           nmp=100,
                           nmax=500,
                           ncmax=(10,))

if __name__=='__main__':
    main()
