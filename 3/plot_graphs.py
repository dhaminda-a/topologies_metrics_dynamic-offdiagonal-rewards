#!/usr/bin/python
# coding=utf-8

# Attempts to replicate Sen and Airiau 2007 paper's Figure 1 with SARSA learning algorithm
# Number of actions: 2, Number of states: 1
# Number of agents: 200, number of iterations: 1200
# Dhaminda Abeywickrama
# 15 December 2019

import numpy as np
import matplotlib.pyplot as plt
import random
import secrets
import pickle

import json
import random
import numpy as np


#from matplotlib as pp#.backends.backend_pdf import PdfPages

from os import listdir

if __name__ == "__main__":
    #average_convergence_rate_array = []

    average_run_diversity_array2 = json.loads(open('average_runs_diversity_SD2.json').read())
    #print("printing len(average_run_diversity_array1)", len(average_run_diversity_array1))

    average_run_diversity_array1 = json.loads(open('average_runs_diversity_SD1.json').read())
        # plotting performance using graphs =>

    plt.figure()
    #ax1 = pickle.load(open("k-40.pickle", "rb"))
    plt.plot(average_run_diversity_array2, label = 'Separation Degree 0.96', linewidth=0.75, color = 'green')
    plt.plot(average_run_diversity_array1, label = 'Separation Degree 0.57', linewidth=0.75, color = 'purple')
    plt.ylim(ymin=0)
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Diversity", fontsize=15)
    plt.title("Effects of Separation Degree on Diversity of the Multi-Agent System", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()

    average_runs_norms1_frequency_lc1_SD1_array = json.loads(open('average_runs_norms1_frequency_lc1_SD1.json').read())
    average_runs_norms1_frequency_lc2_SD1_array = json.loads(open('average_runs_norms1_frequency_lc2_SD1.json').read())
    average_runs_norms1_frequency_lc3_SD1_array = json.loads(open('average_runs_norms1_frequency_lc3_SD1.json').read())
    average_runs_norms1_frequency_lc4_SD1_array = json.loads(open('average_runs_norms1_frequency_lc4_SD1.json').read())
    average_runs_norms1_frequency_lc5_SD1_array = json.loads(open('average_runs_norms1_frequency_lc5_SD1.json').read())
    average_runs_norms1_frequency_lc6_SD1_array = json.loads(open('average_runs_norms1_frequency_lc6_SD1.json').read())
    average_runs_norms1_frequency_lc7_SD1_array = json.loads(open('average_runs_norms1_frequency_lc7_SD1.json').read())
    average_runs_norms1_frequency_lc8_SD1_array = json.loads(open('average_runs_norms1_frequency_lc8_SD1.json').read())
    average_runs_norms1_frequency_lc9_SD1_array = json.loads(open('average_runs_norms1_frequency_lc9_SD1.json').read())
    average_runs_norms1_frequency_lc10_SD1_array = json.loads(open('average_runs_norms1_frequency_lc10_SD1.json').read())

    average_runs_norms2_frequency_lc1_SD1_array = json.loads(open('average_runs_norms2_frequency_lc1_SD1.json').read())
    average_runs_norms2_frequency_lc2_SD1_array = json.loads(open('average_runs_norms2_frequency_lc2_SD1.json').read())
    average_runs_norms2_frequency_lc3_SD1_array = json.loads(open('average_runs_norms2_frequency_lc3_SD1.json').read())
    average_runs_norms2_frequency_lc4_SD1_array = json.loads(open('average_runs_norms2_frequency_lc4_SD1.json').read())
    average_runs_norms2_frequency_lc5_SD1_array = json.loads(open('average_runs_norms2_frequency_lc5_SD1.json').read())
    average_runs_norms2_frequency_lc6_SD1_array = json.loads(open('average_runs_norms2_frequency_lc6_SD1.json').read())
    average_runs_norms2_frequency_lc7_SD1_array = json.loads(open('average_runs_norms2_frequency_lc7_SD1.json').read())
    average_runs_norms2_frequency_lc8_SD1_array = json.loads(open('average_runs_norms2_frequency_lc8_SD1.json').read())
    average_runs_norms2_frequency_lc9_SD1_array = json.loads(open('average_runs_norms2_frequency_lc9_SD1.json').read())
    average_runs_norms2_frequency_lc10_SD1_array = json.loads(open('average_runs_norms2_frequency_lc10_SD1.json').read())

    average_runs_norms1_frequency_lc1_SD2_array = json.loads(open('average_runs_norms1_frequency_lc1_SD2.json').read())
    average_runs_norms1_frequency_lc2_SD2_array = json.loads(open('average_runs_norms1_frequency_lc2_SD2.json').read())
    average_runs_norms1_frequency_lc3_SD2_array = json.loads(open('average_runs_norms1_frequency_lc3_SD2.json').read())
    average_runs_norms1_frequency_lc4_SD2_array = json.loads(open('average_runs_norms1_frequency_lc4_SD2.json').read())
    average_runs_norms1_frequency_lc5_SD2_array = json.loads(open('average_runs_norms1_frequency_lc5_SD2.json').read())
    average_runs_norms1_frequency_lc6_SD2_array = json.loads(open('average_runs_norms1_frequency_lc6_SD2.json').read())
    average_runs_norms1_frequency_lc7_SD2_array = json.loads(open('average_runs_norms1_frequency_lc7_SD2.json').read())
    average_runs_norms1_frequency_lc8_SD2_array = json.loads(open('average_runs_norms1_frequency_lc8_SD2.json').read())
    average_runs_norms1_frequency_lc9_SD2_array = json.loads(open('average_runs_norms1_frequency_lc9_SD2.json').read())
    average_runs_norms1_frequency_lc10_SD2_array = json.loads(open('average_runs_norms1_frequency_lc10_SD2.json').read())

    average_runs_norms2_frequency_lc1_SD2_array = json.loads(open('average_runs_norms2_frequency_lc1_SD2.json').read())
    average_runs_norms2_frequency_lc2_SD2_array = json.loads(open('average_runs_norms2_frequency_lc2_SD2.json').read())
    average_runs_norms2_frequency_lc3_SD2_array = json.loads(open('average_runs_norms2_frequency_lc3_SD2.json').read())
    average_runs_norms2_frequency_lc4_SD2_array = json.loads(open('average_runs_norms2_frequency_lc4_SD2.json').read())
    average_runs_norms2_frequency_lc5_SD2_array = json.loads(open('average_runs_norms2_frequency_lc5_SD2.json').read())
    average_runs_norms2_frequency_lc6_SD2_array = json.loads(open('average_runs_norms2_frequency_lc6_SD2.json').read())
    average_runs_norms2_frequency_lc7_SD2_array = json.loads(open('average_runs_norms2_frequency_lc7_SD2.json').read())
    average_runs_norms2_frequency_lc8_SD2_array = json.loads(open('average_runs_norms2_frequency_lc8_SD2.json').read())
    average_runs_norms2_frequency_lc9_SD2_array = json.loads(open('average_runs_norms2_frequency_lc9_SD2.json').read())
    average_runs_norms2_frequency_lc10_SD2_array = json.loads(open('average_runs_norms2_frequency_lc10_SD2.json').read())

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_SD1_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_SD1_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_SD1_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_SD1_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_SD1_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_SD1_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_SD1_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_SD1_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_SD1_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_SD1_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_SD1_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_SD1_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_SD1_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_SD1_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_SD1_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_SD1_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_SD1_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_SD1_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_SD1_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_SD1_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

    #plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.show()
    #ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax1.legend(loc = 'upper right', fontsize = 8)
    ax2.legend(loc = 'upper right', fontsize = 8)
    ax3.legend(loc = 'upper right', fontsize = 8)
    ax4.legend(loc = 'upper right', fontsize = 8)
    ax5.legend(loc = 'upper right', fontsize = 8)
    ax6.legend(loc = 'upper right', fontsize = 8)
    ax7.legend(loc = 'upper right', fontsize = 8)
    ax8.legend(loc = 'upper right', fontsize = 8)
    ax9.legend(loc = 'upper right', fontsize = 8)
    ax10.legend(loc = 'upper right', fontsize = 8)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)
    ax9.grid(True)
    ax10.grid(True)

    ax1.set_title('SD1 - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_SD2_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_SD2_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_SD2_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_SD2_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_SD2_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_SD2_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_SD2_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_SD2_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_SD2_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_SD2_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_SD2_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_SD2_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_SD2_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_SD2_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_SD2_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_SD2_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_SD2_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_SD2_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_SD2_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_SD2_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

    #plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.show()
    #ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax1.legend(loc = 'upper right', fontsize = 8)
    ax2.legend(loc = 'upper right', fontsize = 8)
    ax3.legend(loc = 'upper right', fontsize = 8)
    ax4.legend(loc = 'upper right', fontsize = 8)
    ax5.legend(loc = 'upper right', fontsize = 8)
    ax6.legend(loc = 'upper right', fontsize = 8)
    ax7.legend(loc = 'upper right', fontsize = 8)
    ax8.legend(loc = 'upper right', fontsize = 8)
    ax9.legend(loc = 'upper right', fontsize = 8)
    ax10.legend(loc = 'upper right', fontsize = 8)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)
    ax9.grid(True)
    ax10.grid(True)

    ax1.set_title('SD2 - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
    

