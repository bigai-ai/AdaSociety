import gurobipy as gp
from gurobipy import GRB
from project.utils.config_loader import ConfigLoader
import numpy as np
from itertools import product
import warnings
import networkx as nx

def is_dag(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(G)

def cal_oracle(config_dir='./config/main.json'):
    config_loader = ConfigLoader(config_dir)
    config = config_loader.config
    event_lib = config.event
    res2num = {}
    events = set()
    jobs = set()
    for resource in config.task.static.resources:
        if resource["name"] not in res2num:
            res2num[resource["name"]] = 0
        num = resource["num"] if isinstance(resource["num"], int) else sum(resource["num"])
        res2num[resource["name"]] += num
    for resource in config.task.random.resources:
        if resource["name"] not in res2num:
            res2num[resource["name"]] = 0
        res2num[resource["name"]] += resource["num"]["max"] * resource["repeat"]

    for event in config.task.static.events:
        events.add(event["name"])
    for event in config.task.random.events:
        events.add(event["name"])

    for player in config.task.static.players:
        jobs.add(player["job"])
    for player in config.task.random.players:
        jobs.add(player["job"])

    edges = []
    for name in events:
        for resource in event_lib[name]['in']:
            if resource not in res2num:
                res2num[resource] = 0
        for resource in event_lib[name]['out']:
            if resource not in res2num:
                res2num[resource] = 0
        event_req = event_lib[name].get("requirements", {})
        start = list(event_req.keys()) + list(event_lib[name]['in'].keys())
        edges.extend(product(start, event_lib[name]['out'].keys()))

    resources = []
    res_num = []
    res2value = {res: 0 for res in res2num}
    for res, num in res2num.items():
        resources.append(res)
        res_num.append(num)

    for name in resources:
        res_req = config.resource[name].get("requirements", {})
        edges.extend(product(res_req.keys(), [name]))
    
    if not is_dag(resources, edges):
        warnings.warn("The resource dependency graph is not a directed acyclic graph, where the algorithm will fail")

    for name in jobs:
        for resource, score in config.job[name]["inventory"]["score"].items():
            if score > res2value[resource]:
                res2value[resource] = score

    events = list(events)
    n = len(resources)
    res2id = {res: i for i, res in enumerate(resources)}
    in_mat = np.zeros((len(events), n))
    out_mat = np.zeros((len(events), n))
    for i, name in enumerate(events):
        for res, num in event_lib[name]['in'].items():
            in_mat[i, res2id[res]] = num
        for res, num in event_lib[name]['out'].items():
            out_mat[i, res2id[res]] = num
    io_mat = out_mat - in_mat
    values = [res2value[res] for res in resources]

    model = gp.Model('resouce_optimization')

    # Decision Variables
    x = model.addVars(len(events), vtype=GRB.INTEGER, name='x') # events times
    r = model.addVars(n, lb=0, vtype=GRB.INTEGER, name='r') # left resource
    d = model.addVars(len(events), vtype=GRB.BINARY, name='d') # events occured
    e = model.addVars(n, vtype=GRB.BINARY, name='e') # resources occured

    # Objective
    obj = gp.quicksum(r[i]*values[i] for i in range(n))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    M = sum(res_num) * np.maximum(1, np.max(np.sum(io_mat, axis=1)))
    model.addConstrs((r[i] == res_num[i] + gp.quicksum(x[j]*io_mat[j][i] for j in range(len(events))) for i in range(n)), name='resource_left')
    model.addConstrs((x[j] <= d[j] * M for j in range(len(events))), name='event_occurred')
    for event_id, name in enumerate(events):
        event_req = config.event[name].get("requirements", {})
        for req, threshold in event_req.items():
            model.addConstr(res_num[res2id[req]] + gp.quicksum(x[j]*out_mat[j][res2id[req]] for j in range(len(events))) >= threshold * d[event_id], name='requirement')
    for name in resources:
        res_req = config.resource[name].get("requirements", {})
        for req, threshold in res_req.items():
            model.addConstr(res_num[res2id[req]] + gp.quicksum(x[j]*out_mat[j][res2id[req]] for j in range(len(events))) >= threshold * e[res2id[name]], name='requirement')
    in_relation = np.where(in_mat > 0)
    for i, j in zip(in_relation[0], in_relation[1]):
        model.addConstr(d[i] <= e[j], name='event_dependency')

    model.optimize()

    if model.status == GRB.OPTIMAL:
        # x_values = model.getAttr('x', x)
        # r_values = model.getAttr('x', r)
        # d_values = model.getAttr('x', d)
        # e_values = model.getAttr('x', e)
        # print(x_values)
        # print(r_values)
        # print(d_values)
        # print(e_values)
        return model.ObjVal
    else:
        warnings.warn("No optimal solution found, return 1")
        return 1
    
if __name__ == '__main__':
    print(cal_oracle('./config/main.json'))