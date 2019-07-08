# -*- coding: utf-8 -*-
"""
@author: Yin Wan
@environment: Python 3.7
@file: flow_map_arrow.py
@time: 2019/5/26 
"""
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def count_trajectory(csv_name):
    df = pd.read_csv(csv_name, encoding='utf-8')
    dfcount = df.groupby(['start_point_bbox', 'end_point_bbox'], as_index=False)['start_point'].count()
    dfcount = dfcount.rename(columns={'start_point': 'count'})
    dfcount = dfcount.sort_values(by=['count'], ascending=False)
    # dfcount.to_csv(csv_name, index=0, encoding='utf8')
    return dfcount


def bounding_box2lon_lat(row):
    lst_of_coords = eval(row)
    longitude = float(lst_of_coords[0])
    latitude = float(lst_of_coords[1])
    return [longitude, latitude]


def draw_arrow(x_begin, y_begin, x_end, y_end, color, width, alpha):
    plt.arrow(x_begin, y_begin, x_end - x_begin, y_end - y_begin,
              length_includes_head=True, linewidth=width,
              head_width=0.5,  head_length=0.5, fc=color, ec=color, alpha = alpha)


def draw_map(df, map, col, widths=[1, 5, 9, 13], alpha=0.8):
    for i in range(df.iloc[:, 0].size):
        start = df.loc[i, 'start_point_bbox']
        end = df.loc[i, 'end_point_bbox']
        count = df.loc[i, 'count']
        c = int(count/10)
        start_list = bounding_box2lon_lat(start)
        end_list = bounding_box2lon_lat(end)
        x1, y1 = map(start_list[0], start_list[1])
        x2, y2 = map(end_list[0], end_list[1])
        if x1 != x2 and y1 != y2:
            map.plot(x1, y1, 'bo', markersize=2, alpha=.3, color=col)
            l, = map.plot(x2, y2, 'bo', markersize=2, alpha=.3, color=col)
            draw_arrow(x1, y1, x2, y2, col, widths[c], alpha)
        if i % 100 == 0:
            print("%d / %d user_ids have been completed." % (i, df.iloc[:, 0].size))
    return l


def plot_recommend_place(df, map, col):
    for i in range(df.iloc[:, 0].size):
        bounding_box = df.loc[i, 'bbox']
        score = df.loc[i, 'score']
        c = int(score*20)

        lon_lat = bounding_box2lon_lat(bounding_box)
        x, y = map(lon_lat[0], lon_lat[1])
        l, = map.plot(x, y, '*', markersize=c, color=col)

        if i % 100 == 0:
            print("%d / %d places have been completed." % (i, df.iloc[:, 0].size))
    return l


if __name__ == "__main__":

    map = Basemap(llcrnrlon=-140.25,  # lower left corner longitude of contiguous US
                  llcrnrlat=5.0,  # lower left corner latitude of contiguous US
                  urcrnrlon=-56.25,  # upper right corner longitude of contiguous US
                  urcrnrlat=54.75,  # upper right corner latitude of contiguous US
                  epsg=4269)
    # http://server.arcgisonline.com/arcgis/rest/services
    # EPSG Number of America is 4269
    map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=1500, verbose=True)
    '''
    map = Basemap(projection='merc',
                  resolution='h',
                  area_thresh=10000,
                  llcrnrlon=-140.25,  # lower left corner longitude of contiguous US
                  llcrnrlat=5.0,  # lower left corner latitude of contiguous US
                  urcrnrlon=-56.25,  # upper right corner longitude of contiguous US
                  urcrnrlat=54.75)  # upper right corner latitude of contiguous US
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    map.fillcontinents(color='#DAF7A6', lake_color='#a7cdf2')
    map.drawmapboundary(fill_color='#a7cdf2')
    '''

    df = count_trajectory('i104249727_finaldata.csv')
    df_f = count_trajectory('frds_i104249727_finaldata.csv')
    df_ff = count_trajectory('foff_i104249727_finaldata.csv')
    df_place = pd.read_csv('i104249727_want_to_go.csv', encoding='utf-8')

    l3 = draw_map(df_ff, map, 'orange')
    l2 = draw_map(df_f, map, 'cyan')
    l1 = draw_map(df, map, 'red', widths=[2, 10, 18, 26])

    l4 = plot_recommend_place(df_place, map, 'magenta')

    plt.gcf().set_size_inches(15, 15)

    plt.legend(handles=[l1, l2, l3, l4,],
               labels=['User 104249727',
                       'User 104249727\'s friends',
                       'Friends of user 104249727\'s friends',
                       'Recommended places for user 104249727'],
               loc='lower left', fontsize='xx-large')

    plt.savefig('flow_map_weight_104249727&f.png', format='png', dpi=1000)
