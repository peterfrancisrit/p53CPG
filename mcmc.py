# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pymc3 as pm
from collections import OrderedDict
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import BoxAnnotation
import itertools


''' INITIALISE PARAMATERS '''
window = 50 # tested a few, this seemed quite good
n_samples = 10000 # 5000 was a base
filename = ["p53.fasta", "p53_zebra.fasta","p53_platy.fasta"]
data = OrderedDict()


def read_file(filename):
    '''
    Read the fasta files
    param: filename -> filename (str)
    '''

    with open(filename) as file:
        header = " "
        sequence = " "
        for line in file:
            if ">" in line[0]:
                header += line
            else:
                if " " not in line:
                    sequence += line

    return header, sequence

def window_cg(sequence,window_length):
    '''
    Calculate CpG count for a given window
    param: sequence -> fasta sequence (str)
    param: window_length -> length of window
    '''
    content = []
    offset = 0
    for i in range(0,len(sequence) - window_length,window_length):
        sub_string = sequence[i+offset:i+window_length]
        offset = 1
        content.append(sub_string.count('CG'))
    return np.array(content)

def rolling_window_cg(sequence,window_length):
    '''
    Calculate rolling CpG count for a given window
    param: sequence -> fasta sequence (str)
    param: window_length -> length of window
    '''
    assert(len(sequence) > window_length)
    print(len(sequence) - window_length)
    content = []
    for i in range(len(sequence) - window_length):
        sub_string = sequence[i:i+window_length]
        content.append(sub_string.count('CG'))
    return np.array(content)

def model(gene,n_samples):
    '''
    The PyMC3 model following the form of
    D(r) ~ Poisson(r) => r = {initial if g_min < s else second}
    s ~ Uniform(g_min, g_max)
    initial ~ Exponential(1)
    second ~ Exponential(1)

    param: gene -> count data in np.array()
    param: n_samples -> number of sampling procedures
    '''

    data = pd.DataFrame()
    data['val'] = gene

    with pm.Model() as GCModel:

        switch = pm.DiscreteUniform('switch', lower=min(data.index), upper=max(data.index), testval=data.val.mean())

        # Priors for pre- and post-switch rates number of disasters
        first = pm.HalfCauchy('initial',1)
        second = pm.HalfCauchy('second',1)

        # Allocate appropriate Poisson rates to years before and after current
        change = pm.math.switch(switch >= data.index, first, second)

        model = pm.Poisson('model', change, observed=data.val.values)

        trace = pm.sample(n_samples,tune=5000)
        return trace

def run_model(data):
    result = [model(data[d][0],n_samples) for d in data]
    for i in result:
        pm.traceplot(i)
        pm.autocorrplot(i)
    plt.show()

def plot_windows(data):
    '''
    Plotting the windows in terms of counts with "hypothesise split"
    param: data -> ([array of counts], [array of range length])
    '''

    output_file("scatter.html")
    p = figure(title="CpG Content of p53",)
    p.background_fill_color = "grey"
    p.background_fill_alpha = 0.2
    p.add_layout(BoxAnnotation(left=40, right=160,fill_alpha=0.2, fill_color='blue'))
    colors = itertools.cycle(palette)
    for i , color in zip(data, colors):
        if i == 'Platypus':
            pp = figure()
            pp.circle(data[i][1],data[i][0],alpha=0.5,legend_label=i,color=color)
            pp.background_fill_color = "grey"
            pp.background_fill_alpha = 0.2
            pp.add_layout(BoxAnnotation(left=5, right=12,fill_alpha=0.2, fill_color='blue'))
        else:
            p.circle(data[i][1],data[i][0],alpha=0.8,color=color,legend_label=i)
    show(row(p,pp))

def plot_histogram(data):
    '''
    Plotting the histogram
    param: data -> ([array of count], [array of range len])
    '''
    output_file("histogram.html")
    p = figure(title="CpG Content of p53",width=400,height=400)
    p.background_fill_color = "grey"
    p.background_fill_alpha = 0.2
    colors = itertools.cycle(palette)
    for i , color in zip(data, colors):
        hist, edges = np.histogram(data[i][0], density=True)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           color=color, alpha=0.5,legend_label=i)
    show(p)


def plot_final(gene,trace,label):
    '''
    Function to visualise the switch point from the sample
    param: gene -> counts of cpg form np.array()
    param: trace -> the trace from the fitted PyMC3 Model
    param: label -> string for plot label name
    '''

    # create empty dataframe
    data = pd.DataFrame()
    data['val'] = gene

    plt.figure(figsize=(10, 8))
    plt.title("CpG Count of {} p53".format(label))
    plt.plot(data.index, data['val'], '.')
    plt.ylabel("Normalized CpG Count ", fontsize=16)
    plt.xlabel("Windows | window length = {}".format(window), fontsize=16)

    plt.vlines(trace['switch'].mean(), data['val'].min(), data['val'].max(), color='black')
    average_cpg = np.zeros_like(data['val'], dtype='float')
    for i, sub in enumerate(data.index):
        idx = sub < trace['switch']
        average_cpg[i] = (trace['initial'][idx].sum() + trace['second'][~idx].sum()) / (len(trace) * trace.nchains)

    sp_hpd = pm.hpd(trace['switch'])
    plt.fill_betweenx(y=[data.val.min(), data.val.max()],
                      x1=sp_hpd[0], x2=sp_hpd[1], alpha=0.2, color='black');
    plt.plot(data.index, average_cpg,  'k--', lw=2,color='r');

''' READ '''
header_human, sequence_human = read_file(filename[0])
header_fish, sequence_fish = read_file(filename[1])
header_plat, sequence_plat = read_file(filename[2])

''' COUNT '''
data['Human'] = (window_cg(sequence_human, window), list(range(len(window_cg(sequence_human, window)))))
data['Zebrafish'] = (window_cg(sequence_fish, window), list(range(len(window_cg(sequence_fish, window)))))
data['Platypus'] = (window_cg(sequence_plat, window), list(range(len(window_cg(sequence_plat,window)))))


''' INITIAL PLOTTING '''

plot_windows(data)
plot_histogram(data)

''' MODELLING '''
# run_model(data)
# ''' FINAL PLOTTING '''
# plot_final(cg_human,trace_human,"Human")
# plot_final(cg_fish,trace_fish,"Zebrafish")
# plot_final(cg_plat,trace_plat,"Platypus")
#
# plt.show()
