---
layout: project
title: "Animated Line Plot with Matplotlib"
description: Making an animated line plot in Matplotlib
category: Visualization
---

Code for making an animated line plot in Matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load in data
df = pd.read_csv('hockey_stats.csv')
df.set_index('Age', inplace=True)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(top=.85, bottom=.15)

# Initialize empty line objects
line1, = ax.plot([], [], label='Crosby', color='k')
line2, = ax.plot([], [], label='McDavid', color='#FF4C00')

# Setting up the plot limits
ax.set_xlim(min(df.index), max(df.index))
ax.set_ylim(min(df.min()) - 10, 2000)

# Adding labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Total Points')
ax.legend()
ax.set_xticks(df.index, [str(x) for x in df.index])
ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)
ax.text(x = 16.5, y = -400,
    s = '     SikFlow.io                                                                                                                                       Source: NHL.com         ',
        fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')
ax.text(x=17, y=2300,s='Total Points by Age - Sidney Crosby vs Connor McDavid',
        fontsize = 26, weight = 'bold', alpha = .75)
ax.text(x=17, y=2150, s='After their age 27 season, Sidney Crosby had 853 total points while Connor McDavid has 982 total points')

# Initialize the data arrays
xdata, ydata1, ydata2 = [], [], []

# Function to initialize the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Function to update the animation
def update(frame):
    xdata.append(frame)
    ydata1.append(df['Crosby'][frame])
    ydata2.append(df['McDavid'][frame])

    line1.set_data(xdata, ydata1)
    line2.set_data(xdata, ydata2)
    
    return line1, line2

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=df.index, init_func=init,
                              blit=True, repeat=False)

# Save the animation as a GIF
ani.save('crosby_vs_mcdavid.gif', writer='pillow', fps=2)

# Save the animation as an MP4
#ani.save('crosby_vs_mcdavid.mp4', writer='ffmpeg', fps=2)
```

![Animated Line Plot](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_posts/Images/crosby_vs_mcdavid.gif)