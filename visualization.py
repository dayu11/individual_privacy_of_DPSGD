import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sess', type=str, default='resnet20_cifar10', help='session name')



args = parser.parse_args()

os.makedirs('figs', exist_ok=True)

### Plot the histogram of the individual privacy loss
# Load the individual privacy loss
idp_path = 'stats/' + args.sess + '_privacy_profile.npy'
idp_parameters = np.load(idp_path)
# Get the axis
fig, ax = plt.subplots()
# Plot the histogram
ax.hist(idp_parameters, bins=20 , alpha=0.5, color='b')
# Set the title
ax.set_title('Histogram of the individual privacy loss')
# Set the x-axis label
ax.set_xlabel(r'$\varepsilon$')
# Set the y-axis label
ax.set_ylabel('Counts')
# Save the figure
fig.savefig('figs/' + args.sess + '_histogram.png')
print('histogram of individual privacy saved to figs/' + args.sess + '_histogram.png')
# clear the figure
plt.clf()

### Plot the estimation error
#Load ground truth
gt_path = 'stats/' + args.sess + '_ghost_privacy_profile.npy'
gt_parameters = np.load(gt_path)[0:1000]
estimated_parameters = idp_parameters[0:1000]
# Get the axis
fig, ax = plt.subplots()
# Compute the pearson correlation coefficient
corr = np.corrcoef(gt_parameters, estimated_parameters)[0,1]
# Plot the y=x line
ax.plot(np.arange(0, 8), np.arange(0, 8), color='g', label='y=x')
# Make the scatter plot
ax.scatter(gt_parameters, estimated_parameters, color='b', label='estimated', s=0.3)
# Add a text box at the lower right corner that shows the pearson correlation coefficient
ax.text(0.5, 0.1, 'Pearsons\' r: {:.3f}'.format(corr), transform=ax.transAxes, fontsize=14, verticalalignment='top')
# Set the title
ax.set_title(r'Real $\varepsilon$ vs estimated $\varepsilon$')
# Set the x-axis label
ax.set_xlabel(r'Real $\varepsilon$')
# Set the y-axis label
ax.set_ylabel(r'Estimated $\varepsilon$')
# Set the legend
ax.legend()
# Save the figure
fig.savefig('figs/' + args.sess + '_estimation_error.png')
print('estimation error saved to figs/' + args.sess + '_estimation_error.png')
# clear the figure
plt.clf()



