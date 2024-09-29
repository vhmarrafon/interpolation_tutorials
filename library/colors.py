import logging

from colour import Color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class ColorScale:

    def __init__(self):

        self.colors = {}
        self._default_colors()

    def _default_colors(self):

        default = {
            # blue to red
            'c1': ['#00263E', '#1F8ACB', '#80FFFD', '#0A6501', "#6EA739", "#98A739", "#FFE618",
                   '#FFC318', "#FF9218", "#FF5E18", "#FF1818", "#890000"],
            
            'c1_r': ['#00263E', '#1F8ACB', '#80FFFD', '#0A6501', "#6EA739", "#98A739", "#FFE618",
                   '#FFC318', "#FF9218", "#FF5E18", "#FF1818", "#890000"][::-1],

            # white to indigo
            'c2': ['#FFFFFF', '#BDC9CC', '#729BA5', '#54A2B5', '#63FB91', '#D9FB63', '#FFF700',
                   '#FFA200', '#FF4200', '#8C0077']
        }

        for n, c in default.items():
            self.colors[n] = self.define_cmap(c, name=n)

    @staticmethod
    def define_cmap(colors, name='my_list', show=False):
        '''
        :param name str:
        :param colors list: list of hexadecimal colors
        '''

        color_ramp = LinearSegmentedColormap.from_list(name, [Color(c1).rgb for c1 in colors])

        if show:
            plt.figure(figsize=(15, 3))
            plt.imshow([list(np.arange(0, len(colors), 0.1))], interpolation='nearest', origin='lower',
                       cmap=color_ramp)
            plt.xticks([])
            plt.yticks([])

        return color_ramp

    def make_colors(self, conf={}):
        '''
        method to create a new color

        :param conf dict: {'name': ['hexadecimal colors']}
        '''

        if (not conf) or (not isinstance(conf, dict)):
            logging.error(
                'To make a color you must pass a dict @ {"color_name": ["color1, color2..."]} colors in hexadecimal!')
            return

        # creating colors
        for n, cols in conf.items():
            self.colors[n] = self.define_cmap(cols, name=n)

    def cmap(self, name):

        if name not in self.colors.keys():
            return plt.get_cmap(name)

        else:
            return self.colors[name]

