# DASMC - [module description]
# Copyright (C) 2026 Your Name / Your Institution
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from ete3 import Tree

#Based on the idea of Xie & Zhang (2023) [ARTree]
def TreeToVector(tree):
    sumt_tree=tree.copy()
    i=0
    node_list=[]
    prune_list=[]
    regraft_list=[]
    no_node_dict={}
    name_list=sorted(sumt_tree.get_leaf_names())
    for name in name_list:
        node_list.append(sumt_tree.get_leaves_by_name(name)[0])
        i+=1

    center_node1=node_list[0].get_common_ancestor([node_list[1]])
    center_node2=node_list[0].get_common_ancestor([node_list[2]])
    center_node3=node_list[1].get_common_ancestor([node_list[2]])
    if center_node1 in center_node2.get_ancestors():
        if center_node2 in center_node3.get_ancestors():
            center_node=center_node3
        else:
            center_node = center_node2
    else:
        if center_node3 in center_node1.get_ancestors():
            center_node = center_node1
        else:
            center_node = center_node3
    center_node.name='center'
    center_node.add_feature('no',0)
    no_node_dict[0]=center_node
    i=1
    for node in node_list[:3]:
        node.add_feature('no',i)
        no_node_dict[i]=node
        i+=1

    try:
        sumt_tree.set_outgroup(center_node)
    except Exception as e:
        e=0

    n=len(node_list)
    i=0
    for node in node_list[:2:-1]:
        father_node=node.up
        sister_node=node.get_sisters()[0]
        grand_node=father_node.up
        if grand_node is None:
            father_node = center_node
            prune_list.append([node, father_node])

            sumt_tree=center_node
            continue

        if grand_node == sumt_tree:
            grand_node = center_node

        node.add_feature('no',n-i)
        no_node_dict[n-i]=node
        father_node.add_feature('no',2*n-3-i)
        no_node_dict[2*n-3-i]=father_node
        prune_list.append([node, father_node])
        regraft_list.append([sister_node,grand_node])
        node.delete(prevent_nondicotomic=False)
        father_node.delete(prevent_nondicotomic=False)
        i+=1

    new_tree_center=sumt_tree
    new_tree_center.add_feature('no',0)
    new_tree=new_tree_center

    vector=[(node_list[0].dist,node_list[1].dist,node_list[2].dist)]
    for n in node_list[3:]:
        sister_node,grand_node=regraft_list.pop()
        grand_node=no_node_dict[grand_node.no]
        sister_node=no_node_dict[sister_node.no]

        node,father_node=prune_list.pop()

        new_father_node=Tree()
        new_father_node.dist=father_node.dist
        new_father_node.add_feature('no',father_node.no)
        no_node_dict[father_node.no]=new_father_node
        new_node=Tree()
        new_node.dist=node.dist
        new_node.add_feature('no',node.no)
        new_node.name=node.name
        no_node_dict[node.no]=new_node
        new_sister_nodes=grand_node.get_children()
        if no_node_dict[sister_node.no] not in new_sister_nodes:
            grand_node=new_tree

        grand_node.remove_child(no_node_dict[sister_node.no])
        new_father_node.add_child(no_node_dict[sister_node.no])
        new_father_node.add_child(new_node)
        grand_node.add_child(new_father_node)
        if grand_node==new_tree:
            vector.append((sister_node.no, new_father_node.dist+new_father_node.get_sisters()[0].dist, new_node.dist))
        else:
            vector.append((sister_node.no, new_father_node.dist, new_node.dist))
    return vector

