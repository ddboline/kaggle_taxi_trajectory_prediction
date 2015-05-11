#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 23:15:29 2015

@author: ddboline
"""
import os
import matplotlib
matplotlib.use('Agg')
import pylab as pl

from pandas.tools.plotting import scatter_matrix

def create_html_page_of_plots(list_of_plots, prefix='html'):
    """
    create html page with png files
    """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    os.system('mv *.png %s' % prefix)
    #print(list_of_plots)
    idx = 0
    htmlfile = open('%s/index_0.html' % prefix, 'w')
    htmlfile.write('<!DOCTYPE html><html><body><div>\n')
    for plot in list_of_plots:
        if idx > 0 and idx % 200 == 0:
            htmlfile.write('</div></html></html>\n')
            htmlfile.close()
            htmlfile = open('%s/index_%d.html' % (prefix, (idx//200)), 'w')
            htmlfile.write('<!DOCTYPE html><html><body><div>\n')
        htmlfile.write('<p><img src="%s"></p>\n' % plot)
        idx += 1
    htmlfile.write('</div></html></html>\n')
    htmlfile.close()

def plot_data(indf, prefix='html', do_scatter=False):
    """
    create scatter matrix plot, histograms
    """
    list_of_plots = []
    if do_scatter:
        column_groups = []
        for idx in range(0, len(indf.columns), 3):
            print len(indf.columns), idx, (idx+3)
            column_groups.append(indf.columns[idx:(idx+3)])

        for idx in range(len(column_groups)):
            for idy in range(0, idx):
                if idx == idy:
                    continue
                print column_groups[idx]+column_groups[idy]
                pl.clf()
                scatter_matrix(indf[column_groups[idx]+column_groups[idy]])
                pl.savefig('scatter_matrix_%d_%d.png' % (idx, idy))
                list_of_plots.append('scatter_matrix_%d_%d.png' % (idx, idy))
                pl.close()

    for col in indf:
        pl.clf()
        print col
        indf[col].hist(histtype='step', normed=True)
        pl.title(col)
        pl.savefig('%s_hist.png' % col)
        list_of_plots.append('%s_hist.png' % col)

    create_html_page_of_plots(list_of_plots, prefix)
    return
