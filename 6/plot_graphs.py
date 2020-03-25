import numpy as np
import matplotlib.pyplot as plt
import random
import secrets
import pickle
import json

from os import listdir
       
if __name__ == "__main__":
    average_runs_conformity_communitysize50array = json.loads(open('average_runs_conformity_CommunitySize50.json').read())
    average_runs_conformity_communitysize100array = json.loads(open('average_runs_conformity_CommunitySize100.json').read())
    average_runs_conformity_communitysize200array = json.loads(open('average_runs_conformity_CommunitySize200.json').read())

    # plotting performance using graphs =>    
    plt.figure()
    plt.plot(average_runs_conformity_communitysize50array, label = 'Community Size - 50', linewidth=0.75, color = 'red')
    plt.plot(average_runs_conformity_communitysize100array, label = 'Community Size - 100', linewidth=0.75, color = 'gold')
    plt.plot(average_runs_conformity_communitysize200array, label = 'Community Size - 200', linewidth=0.75, color = 'green')
    plt.ylim(ymin=0)
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Conformity", fontsize=15)
    plt.title("Effects of Average Community Size on the Speed of Diverse Local Convention Emergence", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()

    average_runs_conformity_lc1_CommunitySize50_array = json.loads(open('average_conformity_lc1_CommunitySize50.json').read())
    average_runs_conformity_lc2_CommunitySize50_array = json.loads(open('average_conformity_lc2_CommunitySize50.json').read())
    average_runs_conformity_lc3_CommunitySize50_array = json.loads(open('average_conformity_lc3_CommunitySize50.json').read())
    average_runs_conformity_lc4_CommunitySize50_array = json.loads(open('average_conformity_lc4_CommunitySize50.json').read())
    average_runs_conformity_lc5_CommunitySize50_array = json.loads(open('average_conformity_lc5_CommunitySize50.json').read())
    average_runs_conformity_lc6_CommunitySize50_array = json.loads(open('average_conformity_lc6_CommunitySize50.json').read())
    average_runs_conformity_lc7_CommunitySize50_array = json.loads(open('average_conformity_lc7_CommunitySize50.json').read())
    average_runs_conformity_lc8_CommunitySize50_array = json.loads(open('average_conformity_lc8_CommunitySize50.json').read())
    average_runs_conformity_lc9_CommunitySize50_array = json.loads(open('average_conformity_lc9_CommunitySize50.json').read())
    average_runs_conformity_lc10_CommunitySize50_array = json.loads(open('average_conformity_lc10_CommunitySize50.json').read())

    average_runs_conformity_lc1_CommunitySize100_array = json.loads(open('average_conformity_lc1_CommunitySize100.json').read())
    average_runs_conformity_lc2_CommunitySize100_array = json.loads(open('average_conformity_lc2_CommunitySize100.json').read())
    average_runs_conformity_lc3_CommunitySize100_array = json.loads(open('average_conformity_lc3_CommunitySize100.json').read())
    average_runs_conformity_lc4_CommunitySize100_array = json.loads(open('average_conformity_lc4_CommunitySize100.json').read())
    average_runs_conformity_lc5_CommunitySize100_array = json.loads(open('average_conformity_lc5_CommunitySize100.json').read())
    average_runs_conformity_lc6_CommunitySize100_array = json.loads(open('average_conformity_lc6_CommunitySize100.json').read())
    average_runs_conformity_lc7_CommunitySize100_array = json.loads(open('average_conformity_lc7_CommunitySize100.json').read())
    average_runs_conformity_lc8_CommunitySize100_array = json.loads(open('average_conformity_lc8_CommunitySize100.json').read())
    average_runs_conformity_lc9_CommunitySize100_array = json.loads(open('average_conformity_lc9_CommunitySize100.json').read())
    average_runs_conformity_lc10_CommunitySize100_array = json.loads(open('average_conformity_lc10_CommunitySize100.json').read())

    average_runs_conformity_lc1_CommunitySize200_array = json.loads(open('average_conformity_lc1_CommunitySize200.json').read())
    average_runs_conformity_lc2_CommunitySize200_array = json.loads(open('average_conformity_lc2_CommunitySize200.json').read())
    average_runs_conformity_lc3_CommunitySize200_array = json.loads(open('average_conformity_lc3_CommunitySize200.json').read())
    average_runs_conformity_lc4_CommunitySize200_array = json.loads(open('average_conformity_lc4_CommunitySize200.json').read())
    average_runs_conformity_lc5_CommunitySize200_array = json.loads(open('average_conformity_lc5_CommunitySize200.json').read())
    average_runs_conformity_lc6_CommunitySize200_array = json.loads(open('average_conformity_lc6_CommunitySize200.json').read())
    average_runs_conformity_lc7_CommunitySize200_array = json.loads(open('average_conformity_lc7_CommunitySize200.json').read())
    average_runs_conformity_lc8_CommunitySize200_array = json.loads(open('average_conformity_lc8_CommunitySize200.json').read())
    average_runs_conformity_lc9_CommunitySize200_array = json.loads(open('average_conformity_lc9_CommunitySize200.json').read())
    average_runs_conformity_lc10_CommunitySize200_array = json.loads(open('average_conformity_lc10_CommunitySize200.json').read())

    number_of_iterations = 2000
    p = np.arange(0, number_of_iterations, 1)
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(p, average_runs_conformity_lc1_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #1', color='green')
    plt.ylim(ymin=0)
    ax2.plot(p, average_runs_conformity_lc2_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #2', color='red')
    plt.ylim(ymin=0)
    ax3.plot(p, average_runs_conformity_lc3_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #3', color='blue')
    plt.ylim(ymin=0)
    ax4.plot(p, average_runs_conformity_lc4_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #4', color='orange')
    plt.ylim(ymin=0)
    ax5.plot(p, average_runs_conformity_lc5_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #5', color='pink')
    plt.ylim(ymin=0)
    ax6.plot(p, average_runs_conformity_lc6_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #6', color='brown')
    plt.ylim(ymin=0)
    ax7.plot(p, average_runs_conformity_lc7_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #7', color='purple')
    plt.ylim(ymin=0)
    ax8.plot(p, average_runs_conformity_lc8_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #8', color='black')
    plt.ylim(ymin=0)
    ax9.plot(p, average_runs_conformity_lc9_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #9', color='silver')
    plt.ylim(ymin=0)
    ax10.plot(p, average_runs_conformity_lc10_CommunitySize50_array, label = 'CommunitySize50 - Conformity - local community #10', color='yellow')
    plt.ylim(ymin=0)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()
    ax9.legend()
    ax10.legend()
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
    ax1.set_title('CommunitySize50 - Plotting Individual Conformity of all Local Communities (Average in 100 runs)', fontsize=20)
    ax9.set_xlabel('Iterations', fontsize=15)
    ax4.set_ylabel('Conformity', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

    number_of_iterations = 2000
    p = np.arange(0, number_of_iterations, 1)
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(p, average_runs_conformity_lc1_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #1', color='green')
    plt.ylim(ymin=0)
    ax2.plot(p, average_runs_conformity_lc2_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #2', color='red')
    plt.ylim(ymin=0)
    ax3.plot(p, average_runs_conformity_lc3_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #3', color='blue')
    plt.ylim(ymin=0)
    ax4.plot(p, average_runs_conformity_lc4_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #4', color='orange')
    plt.ylim(ymin=0)
    ax5.plot(p, average_runs_conformity_lc5_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #5', color='pink')
    plt.ylim(ymin=0)
    ax6.plot(p, average_runs_conformity_lc6_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #6', color='brown')
    plt.ylim(ymin=0)
    ax7.plot(p, average_runs_conformity_lc7_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #7', color='purple')
    plt.ylim(ymin=0)
    ax8.plot(p, average_runs_conformity_lc8_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #8', color='black')
    plt.ylim(ymin=0)
    ax9.plot(p, average_runs_conformity_lc9_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #9', color='silver')
    plt.ylim(ymin=0)
    ax10.plot(p, average_runs_conformity_lc10_CommunitySize100_array, label = 'CommunitySize100 - Conformity - local community #10', color='yellow')
    plt.ylim(ymin=0)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()
    ax9.legend()
    ax10.legend()
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
    ax1.set_title('CommunitySize100 - Plotting Individual Conformity of all Local Communities (Average in 100 runs)', fontsize=20)
    ax9.set_xlabel('Iterations', fontsize=15)
    ax4.set_ylabel('Conformity', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

    number_of_iterations = 2000
    p = np.arange(0, number_of_iterations, 1)
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(p, average_runs_conformity_lc1_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #1', color='green')
    plt.ylim(ymin=0)
    ax2.plot(p, average_runs_conformity_lc2_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #2', color='red')
    plt.ylim(ymin=0)
    ax3.plot(p, average_runs_conformity_lc3_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #3', color='blue')
    plt.ylim(ymin=0)
    ax4.plot(p, average_runs_conformity_lc4_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #4', color='orange')
    plt.ylim(ymin=0)
    ax5.plot(p, average_runs_conformity_lc5_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #5', color='pink')
    plt.ylim(ymin=0)
    ax6.plot(p, average_runs_conformity_lc6_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #6', color='brown')
    plt.ylim(ymin=0)
    ax7.plot(p, average_runs_conformity_lc7_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #7', color='purple')
    plt.ylim(ymin=0)
    ax8.plot(p, average_runs_conformity_lc8_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #8', color='black')
    plt.ylim(ymin=0)
    ax9.plot(p, average_runs_conformity_lc9_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #9', color='silver')
    plt.ylim(ymin=0)
    ax10.plot(p, average_runs_conformity_lc10_CommunitySize200_array, label = 'CommunitySize200 - Conformity - local community #10', color='yellow')
    plt.ylim(ymin=0)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()
    ax9.legend()
    ax10.legend()
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
    ax1.set_title('CommunitySize200 - Plotting Individual Conformity of all Local Communities (Average in 100 runs)', fontsize=20)
    ax9.set_xlabel('Iterations', fontsize=15)
    ax4.set_ylabel('Conformity', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
    
    average_runs_norms1_frequency_lc1_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc1_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc2_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc2_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc3_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc3_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc4_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc4_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc5_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc5_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc6_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc6_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc7_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc7_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc8_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc8_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc9_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc9_CommunitySize50.json').read())
    average_runs_norms1_frequency_lc10_CommunitySize50_array = json.loads(open('average_runs_norms1_frequency_lc10_CommunitySize50.json').read())

    average_runs_norms2_frequency_lc1_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc1_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc2_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc2_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc3_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc3_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc4_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc4_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc5_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc5_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc6_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc6_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc7_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc7_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc8_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc8_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc9_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc9_CommunitySize50.json').read())
    average_runs_norms2_frequency_lc10_CommunitySize50_array = json.loads(open('average_runs_norms2_frequency_lc10_CommunitySize50.json').read())

    average_runs_norms1_frequency_lc1_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc1_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc2_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc2_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc3_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc3_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc4_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc4_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc5_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc5_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc6_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc6_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc7_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc7_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc8_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc8_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc9_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc9_CommunitySize100.json').read())
    average_runs_norms1_frequency_lc10_CommunitySize100_array = json.loads(open('average_runs_norms1_frequency_lc10_CommunitySize100.json').read())

    average_runs_norms2_frequency_lc1_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc1_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc2_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc2_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc3_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc3_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc4_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc4_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc5_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc5_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc6_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc6_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc7_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc7_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc8_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc8_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc9_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc9_CommunitySize100.json').read())
    average_runs_norms2_frequency_lc10_CommunitySize100_array = json.loads(open('average_runs_norms2_frequency_lc10_CommunitySize100.json').read())

    average_runs_norms1_frequency_lc1_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc1_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc2_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc2_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc3_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc3_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc4_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc4_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc5_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc5_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc6_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc6_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc7_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc7_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc8_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc8_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc9_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc9_CommunitySize200.json').read())
    average_runs_norms1_frequency_lc10_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc10_CommunitySize200.json').read())

    average_runs_norms2_frequency_lc1_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc1_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc2_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc2_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc3_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc3_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc4_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc4_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc5_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc5_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc6_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc6_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc7_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc7_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc8_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc8_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc9_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc9_CommunitySize200.json').read())
    average_runs_norms2_frequency_lc10_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc10_CommunitySize200.json').read())

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_CommunitySize50_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_CommunitySize50_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_CommunitySize50_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_CommunitySize50_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_CommunitySize50_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_CommunitySize50_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_CommunitySize50_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_CommunitySize50_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_CommunitySize50_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_CommunitySize50_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_CommunitySize50_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_CommunitySize50_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_CommunitySize50_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_CommunitySize50_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_CommunitySize50_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_CommunitySize50_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_CommunitySize50_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_CommunitySize50_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_CommunitySize50_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_CommunitySize50_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

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

    ax1.set_title('CommunitySize50 - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_CommunitySize100_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_CommunitySize100_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_CommunitySize100_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_CommunitySize100_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_CommunitySize100_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_CommunitySize100_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_CommunitySize100_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_CommunitySize100_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_CommunitySize100_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_CommunitySize100_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_CommunitySize100_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_CommunitySize100_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_CommunitySize100_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_CommunitySize100_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_CommunitySize100_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_CommunitySize100_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_CommunitySize100_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_CommunitySize100_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_CommunitySize100_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_CommunitySize100_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

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

    ax1.set_title('CommunitySize100 - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

##    average_runs_norms1_frequency_lc1_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc1_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc2_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc2_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc3_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc3_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc4_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc4_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc5_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc5_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc6_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc6_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc7_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc7_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc8_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc8_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc9_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc9_CommunitySize200.json').read())
##    average_runs_norms1_frequency_lc10_CommunitySize200_array = json.loads(open('average_runs_norms1_frequency_lc10_CommunitySize200.json').read())
##
##    average_runs_norms2_frequency_lc1_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc1_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc2_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc2_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc3_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc3_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc4_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc4_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc5_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc5_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc6_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc6_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc7_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc7_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc8_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc8_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc9_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc9_CommunitySize200.json').read())
##    average_runs_norms2_frequency_lc10_CommunitySize200_array = json.loads(open('average_runs_norms2_frequency_lc10_CommunitySize200.json').read())


    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(average_runs_norms1_frequency_lc1_CommunitySize200_array, label = 'Norm 1 in lc1', linewidth=0.75, color = 'red')
    ax1.plot(average_runs_norms2_frequency_lc1_CommunitySize200_array, label = 'Norm 2 in lc1', linewidth=0.75, color = 'green')

    ax2.plot(average_runs_norms1_frequency_lc2_CommunitySize200_array, label = 'Norm 1 in lc2', linewidth=0.75, color = 'red')
    ax2.plot(average_runs_norms2_frequency_lc2_CommunitySize200_array, label = 'Norm 2 in lc2', linewidth=0.75, color = 'green')

    ax3.plot(average_runs_norms1_frequency_lc3_CommunitySize200_array, label = 'Norm 1 in lc3', linewidth=0.75, color = 'red')
    ax3.plot(average_runs_norms2_frequency_lc3_CommunitySize200_array, label = 'Norm 2 in lc3', linewidth=0.75, color = 'green')

    ax4.plot(average_runs_norms1_frequency_lc4_CommunitySize200_array, label = 'Norm 1 in lc4', linewidth=0.75, color = 'red')
    ax4.plot(average_runs_norms2_frequency_lc4_CommunitySize200_array, label = 'Norm 2 in lc4', linewidth=0.75, color = 'green')

    ax5.plot(average_runs_norms1_frequency_lc5_CommunitySize200_array, label = 'Norm 1 in lc5', linewidth=0.75, color = 'red')
    ax5.plot(average_runs_norms2_frequency_lc5_CommunitySize200_array, label = 'Norm 2 in lc5', linewidth=0.75, color = 'green')

    ax6.plot(average_runs_norms1_frequency_lc6_CommunitySize200_array, label = 'Norm 1 in lc6', linewidth=0.75, color = 'red')
    ax6.plot(average_runs_norms2_frequency_lc6_CommunitySize200_array, label = 'Norm 2 in lc6', linewidth=0.75, color = 'green')

    ax7.plot(average_runs_norms1_frequency_lc7_CommunitySize200_array, label = 'Norm 1 in lc7', linewidth=0.75, color = 'red')
    ax7.plot(average_runs_norms2_frequency_lc7_CommunitySize200_array, label = 'Norm 2 in lc7', linewidth=0.75, color = 'green')

    ax8.plot(average_runs_norms1_frequency_lc8_CommunitySize200_array, label = 'Norm 1 in lc8', linewidth=0.75, color = 'red')
    ax8.plot(average_runs_norms2_frequency_lc8_CommunitySize200_array, label = 'Norm 2 in lc8', linewidth=0.75, color = 'green')

    ax9.plot(average_runs_norms1_frequency_lc9_CommunitySize200_array, label = 'Norm 1 in lc9', linewidth=0.75, color = 'red')
    ax9.plot(average_runs_norms2_frequency_lc9_CommunitySize200_array, label = 'Norm 2 in lc9', linewidth=0.75, color = 'green')

    ax10.plot(average_runs_norms1_frequency_lc10_CommunitySize200_array, label = 'Norm 1 in lc10', linewidth=0.75, color = 'red')
    ax10.plot(average_runs_norms2_frequency_lc10_CommunitySize200_array, label = 'Norm 2 in lc10', linewidth=0.75, color = 'green')

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

    ax1.set_title('CommunitySize200 - Frequence of Norm Emergence (Average in 100 Runs)', fontsize=20)
    ax10.set_xlabel('Iterations', fontsize=15)
    ax5.set_ylabel('Frequence', fontsize=15)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()


