import numpy as np
import matplotlib.pyplot as plt
import random
import secrets
import pickle
import json

from os import listdir
       
if __name__ == "__main__":
    average_runs_diversity_baseline_array = json.loads(open('average_runs_diversity_baseline.json').read())
    #average_runs_diversity_g100_array = json.loads(open('average_runs_diversity_g100.json').read())
    average_runs_diversity_k40_array = json.loads(open('average_runs_diversity_k40.json').read())
    #average_runs_diversity_l20_array = json.loads(open('average_runs_diversity_l20.json').read())

    # plotting performance using graphs =>    
    plt.figure()
    plt.plot(average_runs_diversity_baseline_array, label = 'Baseline (g=50, k=20, l=10)', linewidth=0.75, color = 'red')
    #plt.plot(average_runs_diversity_g100_array, label = 'Community Size g - 100', linewidth=0.75, color = 'gold')
    plt.plot(average_runs_diversity_k40_array, label = 'Average Degree k - 40', linewidth=0.75, color = 'blue')
    #plt.plot(average_runs_diversity_l20_array, label = 'Number of Local Communities l - 20', linewidth=0.75, color = 'blue')

    plt.ylim(ymin=0)
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Diversity", fontsize=15)
    plt.title("Effects of Parameter Change on Diversity Average in 100 runs (Baseline: g=50, k=20, l=10)", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()

    average_runs_norms1_frequency_lc1_baseline_array = json.loads(open('average_runs_norms1_frequency_lc1_baseline.json').read())
    average_runs_norms1_frequency_lc2_baseline_array = json.loads(open('average_runs_norms1_frequency_lc2_baseline.json').read())
    average_runs_norms1_frequency_lc3_baseline_array = json.loads(open('average_runs_norms1_frequency_lc3_baseline.json').read())
    average_runs_norms1_frequency_lc4_baseline_array = json.loads(open('average_runs_norms1_frequency_lc4_baseline.json').read())
    average_runs_norms1_frequency_lc5_baseline_array = json.loads(open('average_runs_norms1_frequency_lc5_baseline.json').read())
    average_runs_norms1_frequency_lc6_baseline_array = json.loads(open('average_runs_norms1_frequency_lc6_baseline.json').read())
    average_runs_norms1_frequency_lc7_baseline_array = json.loads(open('average_runs_norms1_frequency_lc7_baseline.json').read())
    average_runs_norms1_frequency_lc8_baseline_array = json.loads(open('average_runs_norms1_frequency_lc8_baseline.json').read())
    average_runs_norms1_frequency_lc9_baseline_array = json.loads(open('average_runs_norms1_frequency_lc9_baseline.json').read())
    average_runs_norms1_frequency_lc10_baseline_array = json.loads(open('average_runs_norms1_frequency_lc10_baseline.json').read())

    average_runs_norms2_frequency_lc1_baseline_array = json.loads(open('average_runs_norms2_frequency_lc1_baseline.json').read())
    average_runs_norms2_frequency_lc2_baseline_array = json.loads(open('average_runs_norms2_frequency_lc2_baseline.json').read())
    average_runs_norms2_frequency_lc3_baseline_array = json.loads(open('average_runs_norms2_frequency_lc3_baseline.json').read())
    average_runs_norms2_frequency_lc4_baseline_array = json.loads(open('average_runs_norms2_frequency_lc4_baseline.json').read())
    average_runs_norms2_frequency_lc5_baseline_array = json.loads(open('average_runs_norms2_frequency_lc5_baseline.json').read())
    average_runs_norms2_frequency_lc6_baseline_array = json.loads(open('average_runs_norms2_frequency_lc6_baseline.json').read())
    average_runs_norms2_frequency_lc7_baseline_array = json.loads(open('average_runs_norms2_frequency_lc7_baseline.json').read())
    average_runs_norms2_frequency_lc8_baseline_array = json.loads(open('average_runs_norms2_frequency_lc8_baseline.json').read())
    average_runs_norms2_frequency_lc9_baseline_array = json.loads(open('average_runs_norms2_frequency_lc9_baseline.json').read())
    average_runs_norms2_frequency_lc10_baseline_array = json.loads(open('average_runs_norms2_frequency_lc10_baseline.json').read())

    average_runs_norms1_frequency_lc1_k40_array = json.loads(open('average_runs_norms1_frequency_lc1_k40.json').read())
    average_runs_norms1_frequency_lc2_k40_array = json.loads(open('average_runs_norms1_frequency_lc2_k40.json').read())
    average_runs_norms1_frequency_lc3_k40_array = json.loads(open('average_runs_norms1_frequency_lc3_k40.json').read())
    average_runs_norms1_frequency_lc4_k40_array = json.loads(open('average_runs_norms1_frequency_lc4_k40.json').read())
    average_runs_norms1_frequency_lc5_k40_array = json.loads(open('average_runs_norms1_frequency_lc5_k40.json').read())
    average_runs_norms1_frequency_lc6_k40_array = json.loads(open('average_runs_norms1_frequency_lc6_k40.json').read())
    average_runs_norms1_frequency_lc7_k40_array = json.loads(open('average_runs_norms1_frequency_lc7_k40.json').read())
    average_runs_norms1_frequency_lc8_k40_array = json.loads(open('average_runs_norms1_frequency_lc8_k40.json').read())
    average_runs_norms1_frequency_lc9_k40_array = json.loads(open('average_runs_norms1_frequency_lc9_k40.json').read())
    average_runs_norms1_frequency_lc10_k40_array = json.loads(open('average_runs_norms1_frequency_lc10_k40.json').read())

    average_runs_norms2_frequency_lc1_k40_array = json.loads(open('average_runs_norms2_frequency_lc1_k40.json').read())
    average_runs_norms2_frequency_lc2_k40_array = json.loads(open('average_runs_norms2_frequency_lc2_k40.json').read())
    average_runs_norms2_frequency_lc3_k40_array = json.loads(open('average_runs_norms2_frequency_lc3_k40.json').read())
    average_runs_norms2_frequency_lc4_k40_array = json.loads(open('average_runs_norms2_frequency_lc4_k40.json').read())
    average_runs_norms2_frequency_lc5_k40_array = json.loads(open('average_runs_norms2_frequency_lc5_k40.json').read())
    average_runs_norms2_frequency_lc6_k40_array = json.loads(open('average_runs_norms2_frequency_lc6_k40.json').read())
    average_runs_norms2_frequency_lc7_k40_array = json.loads(open('average_runs_norms2_frequency_lc7_k40.json').read())
    average_runs_norms2_frequency_lc8_k40_array = json.loads(open('average_runs_norms2_frequency_lc8_k40.json').read())
    average_runs_norms2_frequency_lc9_k40_array = json.loads(open('average_runs_norms2_frequency_lc9_k40.json').read())
    average_runs_norms2_frequency_lc10_k40_array = json.loads(open('average_runs_norms2_frequency_lc10_k40.json').read())

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_baseline_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_baseline_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_baseline_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_baseline_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_baseline_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_baseline_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_baseline_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_baseline_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_baseline_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_baseline_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_baseline_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_baseline_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_baseline_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_baseline_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_baseline_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_baseline_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_baseline_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_baseline_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_baseline_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_baseline_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

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

    ax1.set_title('baseline - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_k40_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_k40_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_k40_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_k40_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_k40_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_k40_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_k40_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_k40_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_k40_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_k40_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_k40_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_k40_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_k40_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_k40_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_k40_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_k40_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_k40_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_k40_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_k40_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_k40_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

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

    ax1.set_title('k40 - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

