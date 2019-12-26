# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

''' INITIALISE PARAMATERS '''
window = 250 # tested a few, this seemed quite good
n_samples = 5000 # 5000 was a base
filename = ["p53.fasta", "p53_zebra.fasta","p53_platy.fasta"]

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
        first = pm.Exponential('initial',1)
        second = pm.Exponential('second',1)

        # Allocate appropriate Poisson rates to years before and after current
        change = pm.math.switch(switch >= data.index, first, second)

        model = pm.Poisson('model', change, observed=data.val.values)

        trace = pm.sample(n_samples,tune=5000)
        return trace

def plot_windows(gene,fig,ax,label):
    '''
    Plotting the windows in terms of counts with "hypothesise split"
    param: gene -> count data np.array()
    param: fig -> fig from plot
    param: ax -> ax properties from plot
    param: label -> string to name individual plots
    '''

    # Plotting the windows against cpg content and normalized
    data = pd.DataFrame()
    data['x'] = range(len(gene))
    data['y'] = gene / len(gene)

    if label == "Platypus":
        sns.regplot('x','y',data=data,label=label,ax=ax)
    else:
        sns.regplot('x','y',data=data,label=label,ax=ax)
        ax.axvspan(10, 30, alpha=0.05, color='black')
        ax.axvspan(15, 25, alpha=0.1, color='black')
        ax.axvline(20,0,1,ls='--',color='black')


    fig.suptitle("CpG Content of p53 + Estimated Switch (around 20)")
    ax.set_xlabel("Window | window length = {}".format(window))
    ax.set_ylabel("CpG count")
    fig.legend()



def plot_histogram(gene,fig,ax,label):
    '''
    Plotting the histogram
    param: gene -> count data np.array()
    param: fig -> fig properties from plot
    param: ax -> ax properties from plot
    param: label -> string for label name
    '''
    fig.suptitle("CpG Content of p53 | Window = {}".format(window))
    ax = sns.distplot(gene / len(gene),label=label,kde=False)
    ax.set_xlabel(f"CpG count per window = {window}")
    ax.set_ylabel("CpGs count")
    fig.legend()

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

    plt.vlines(trace['switch'].mean(), data['val'].min(), data['val'].max(), color='C1')
    average_cpg = np.zeros_like(data['val'], dtype='float')
    for i, sub in enumerate(data.index):
        idx = sub < trace['switch']
        average_cpg[i] = (trace['initial'][idx].sum() + trace['second'][~idx].sum()) / (len(trace) * trace.nchains)

    sp_hpd = pm.hpd(trace['switch'])
    plt.fill_betweenx(y=[data.val.min(), data.val.max()],
                      x1=sp_hpd[0], x2=sp_hpd[1], alpha=0.5, color='C1');
    plt.plot(data.index, average_cpg,  'k--', lw=2);

''' READ '''
header_human, sequence_human = read_file(filename[0])
header_fish, sequence_fish = read_file(filename[1])
header_plat, sequence_plat = read_file(filename[2])

''' COUNT '''
cg_human = window_cg(sequence_human, window) # let's start with 250
cg_fish = window_cg(sequence_fish, window)
cg_plat = window_cg(sequence_plat, window)

''' INITIAL PLOTTING '''
sns.set_style("dark")
sns.set_palette(["#3498db", "#e74c3c", "#2ecc71"])

fig, ax = plt.subplots(2,1,figsize=(8,8))
plot_windows(cg_human,fig,ax[0],"Human")
plot_windows(cg_fish,fig,ax[0],"Zebrafish")
plot_windows(cg_plat,fig,ax[1],"Platypus")

fig, ax = plt.subplots(figsize=(8,8))
plot_histogram(cg_human,fig,ax,"Human")
plot_histogram(cg_fish,fig,ax,"Zebrafish")
plot_histogram(cg_plat,fig,ax,"Platypus")


''' MODELLING '''
trace_human = model(cg_human,n_samples)
trace_fish = model(cg_fish,n_samples)
trace_plat = model(cg_plat,n_samples)
pm.traceplot(trace_human)
pm.traceplot(trace_fish)
pm.traceplot(trace_plat)
pm.autocorrplot(trace_human, ['switch','initial','second']);
pm.autocorrplot(trace_fish, ['switch','initial','second']);
pm.autocorrplot(trace_plat, ['switch','initial','second']);

''' FINAL PLOTTING '''
plot_final(cg_human,trace_human,"Human")
plot_final(cg_fish,trace_fish,"Zebrafish")
plot_final(cg_plat,trace_plat,"Platypus")

plt.show()
