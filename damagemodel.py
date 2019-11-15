#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : damagemodel.py 
# Creation  : 11 Mar 2016
# Time-stamp: <Sam 2017-06-24 09:47 juergen>
#
# Copyright (c) 2016 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$ 
#
# Description : Damage Model
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
# =============================================================================

from ToolBox import *
import operator
from collections import Counter

class DamageModel(object):
    """
    Damage model for the network.
    The class has the following methods:

        __init__(self, graph,capacity_losses):
            The input parameters:
                graph: A  directed graph of the network.
                capacity_losses

        run()
            This method creates a copy of the graph and sets the non-damaged edges to 0 and
            unassigned objetcs to type 'Road' and uses the capacity_losses values attributed to each
            damage condition and calculates the new capacities.
            In the end creates a dictionary for the damaged edges.

        get_graph()
            returns self.G

        get_damage_dict()
            returns the damage dictionary as damage_dict
    """

    def __init__(self, graph,capacity_losses):
        self.graph = graph
        self.capacity_losses = capacity_losses
        self.damage_dict = {}

    def run(self):
        self.G = self.graph.copy()
        for edge in self.G.edges():
            damage = get_edge_attribute(self.G,edge,'damage')
            object = get_edge_attribute(self.G,edge,'object')

            # set non damaged edges to 0
            if damage == None:
                damage = 0
                set_edge_attribute(self.G,edge,'damage',damage)

            # set not assgind objects to type road
            if object == None:
                object = 'Road'
                set_edge_attribute(self.G,edge,'object',object)

            # change capacity
            initial = get_edge_attribute(self.G,edge,'capacity')
            # round(number, digits), digits: The number of decimals to use when rounding the number. Default is 0
            new = int(round(initial * (1 - self.capacity_losses[object][damage]),0))
            if new == 0:
                new = 1
            set_edge_attribute(self.G,edge,'capacity_i',initial)
            set_edge_attribute(self.G,edge,'capacity',new)

            # assign default restoration parameter
            # this can be probably also done in the restoration model
            set_edge_attribute(self.G,edge,'res_time',0)
            set_edge_attribute(self.G,edge,'res_cost',0)
            set_edge_attribute(self.G,edge,'res_type',0)

            # collect damaged edges and if oneway is true/false. Example:{((x1,y1),(x2,y2)): ['name' , 0]}
            if damage > 0:
                self.damage_dict[edge] = [get_edge_attribute(self.G,edge,'name'),get_edge_attribute(self.G,edge,'oneway')]

    def get_graph(self):
        return self.G

    def get_damage_dict(self,directed=None):
        if directed:
            return self.damage_dict
        else:
            edge_list = []
            for key,value in self.damage_dict.items():
                L = sorted([key[0],key[1]], key=operator.itemgetter(1))
                edge_list.append((L[0],L[1]))

            damage_dict = {}
            counter=Counter(edge_list)

            for key,value in counter.items():
                if value == 1:
                    try:
                        damage_dict[key] = self.damage_dict[key]
                    except KeyError:
                        new_key = (key[1],key[0])
                        damage_dict[new_key] = self.damage_dict[new_key]
                else:
                    damage_dict[key] = self.damage_dict[key]
            return damage_dict

# =============================================================================
# eof
#
# Local Variables: 
# mode: python
# End: 

 
