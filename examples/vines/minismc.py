"""
This example demonstrates the functionality of `pyro.contrib.minipyro`,
which is a minimal implementation of the Pyro Probabilistic Programming
Language that was created for didactic purposes.
"""

from __future__ import absolute_import, division, print_function

import argparse

import torch

# We use the pyro.generic interface to support dynamic choice of backend.
# from pyro.generic import pyro_backend
# from pyro.generic import distributions as dist
# from pyro.generic import infer, optim, pyro

import pyro.contrib.minipyro as pyro
import pyro.distributions as dist

# import pyro.distributions.util as dist_util

from score import score_vine, default_viewport, vv
import six

from ipdb import set_trace as bb
import networkx as nx
import queue
import numpy as np
import math
from render import (
    render_vines, 
    load_target)
# from tkinter import *


def polar2rect(r, theta):
    return r*torch.Tensor([torch.cos(theta), torch.sin(theta)])

class VineNode():
    def __init__(self, pos, is_flower=False):
        self.is_flower = is_flower
        self.pos = pos
        self.is_end = False
    
    def terminal(self):
        return self.is_flower or self.is_end


    # if len(vine.edges()) == 0:
    #     return 

    # from PIL import Image, ImageDraw

    # canvas_width = 250
    # canvas_height = 250
    # white = (255, 255, 255)

    # # PIL create an empty image and draw object to draw on
    # # memory only, not visible
    # image_canvas = Image.new("RGB", (canvas_width, canvas_height), white)
    # draw = ImageDraw.Draw(image_canvas)

    # t_wh = torch.Tensor([canvas_width, canvas_height])
    # s_adjust = torch.Tensor([0, canvas_height])
    # # master = Tk()

    # # w = Canvas(master, 
    # #         width=canvas_width, 
    # #         height=canvas_height)
    # # w.pack()

    # # TODO: normalize the positions to 0, 1?
    # all_pos = torch.cat([n.pos.unsqueeze(0) for n in vine.nodes()])
    # pos_min = all_pos.min(0)[0]
    # pos_range = all_pos.max(0)[0] - pos_min
    # pos_range[pos_range == 0] = 1

    # # go in order of nodes
    # for (in_node, out_node) in vine.edges():
    #     start_pos, end_pos = in_node.pos.detach(), out_node.pos.detach()

    #     # normalize
    #     norm_start = t_wh*(start_pos - pos_min)/pos_range
    #     norm_end = t_wh*(end_pos - pos_min)/pos_range

    #     start_x, start_y = norm_start.cpu().numpy()
    #     end_x, end_y = norm_end.cpu().numpy()
    #     draw.line([start_x, start_y, end_x, end_y], fill="#220000")

    # pix = torch.from_numpy(np.array(image_canvas)).permute(2,0,1)
    # vv.image(pix)


# let's get our model for a simple SMC example
def vine_model(target_img):

    # created a directed graph as our vine holder
    vines = nx.DiGraph()
    start_pos = torch.Tensor([0,0])

    # first we define our root
    root = VineNode(start_pos, is_flower=False)
    
    # add root to our graph
    vines.add_node(root)

    # create our queue
    vine_queue = queue.Queue()

    # add root
    vine_queue.put(root)

    global node_count
    node_count = 0

    def name(prefix="node"):
        global node_count
        return "{}_{}".format(prefix, node_count)

    def add_node(last_node):
        
        r = pyro.sample(
            name("r"),
            dist.Poisson(3)) + 1
        
        theta = pyro.sample(
            name("theta"),
            dist.Normal(0, math.pi/4))

        # which direction to go in 
        delta_pos = polar2rect(r, theta)
        
        # move a little bit, and set no flower
        new_node = VineNode(
                    last_node.pos + delta_pos, 
                    is_flower=False)
        
        # add our node/connection to graph
        vines.add_node(new_node)
        vines.add_edge(last_node, new_node)

        # 
        global node_count
        node_count += 1
        
        # all caught up
        return new_node

    def pyro_factor(vine_obj, target_img):
        # after adding flowers and nodes
        # we render, and set our observe statement
        partial_score, _ = score_vine(vine_obj, target_img)

        # TODO: Hack for now, observe statement
        # returns 1 if we need to exit the computation
        return pyro.sample(name("score"), 
                    dist.Bernoulli(logits=partial_score),
                    obs=torch.ones([1]))
        

    # # we got ourselves a plate y'all
    # with pyro.plate("particles", particle_count, 0):

    # keep generating until death...
    while vine_queue.qsize() > 0:
        
        # get previous node
        last_node = vine_queue.get()

        # we decide if we want a flower or to end
        is_flower = pyro.sample(
            name("flower"),
            dist.Bernoulli(.3))
        
        # potentially, we're a flower!
        # we're a flower!
        last_node.is_flower = is_flower

        # we decide to split on last node
        if not last_node.terminal():
            
            did_add_node = False
            # might split twice! 
            for b_ix in range(2):
                # flip on whether to split or not
                if pyro.sample(
                    name("branch_{}".format(b_ix)),
                    dist.Bernoulli(.5)):
                    
                    # woo-hoo, let's split and add some nodes
                    new_node = add_node(last_node)
                    
                    # now make some choices
                    vine_queue.put(new_node)
                    did_add_node = True
        
            # end of the line bub
            last_node.is_end = not did_add_node

        # here we sample with observation
        # this will be our target dist to match in smc
        if pyro_factor(vines, target_img):
            # if this comes back true, we break
            # TODO: This is a hack for "factor"
            # otherwise, we'd continue generating 
            # and waste cycles -- it's like an abort
            return vines
        
    print("vine model finish")
    # now we've processed all nodes
    return vines

# patching the sample function 
# because we need "is_observed" meta info
def sample_patch(name, fn, obs=None):

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not pyro.PYRO_STACK:
        return fn()

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": (),
        "value": obs,
        "is_observed": obs is not None
    }

    # ...and use apply_stack to send it to the Messengers
    msg = pyro.apply_stack(initial_msg)
    return msg["value"]

# does normal sample, but adds is_observed -- which we use for smc
pyro.sample = sample_patch

# class trace_patch(pyro.trace):
#     # add input/return messages
#     def __call__(self, *args, **kwargs):
#         with self:
#             ret = self.fn(*args, **kwargs)
#             self.trace["_RETURN"] = {"name": "_RETURN", "type":"return", "value": ret}
        
#         return ret

# pyro.trace = trace_patch

def log_prob_sum(trace, site_filter=lambda name, site: True):
    """
    Compute the site-wise log probabilities of the trace.
    Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
    Each ``log_prob_sum`` is a scalar.
    The computation of ``log_prob_sum`` is memoized.
    :returns: total log probability.
    :rtype: torch.Tensor
    """
    result = 0.0
    for name, site in trace.items():
        if site["type"] == "sample" and site_filter(name, site):
            if "log_prob_sum" in site:
                log_p = site["log_prob_sum"]
            else:
                log_p = site["fn"].log_prob(site["value"], *site["args"], **site.get('kwargs', {}))
                log_p = dist.util.scale_and_mask(log_p, site.get("scale", 1.0), site.get("mask", None)).sum()
                site["log_prob_sum"] = log_p
            result = result + log_p
    return result


# https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf
# capture traces, resampling, re-weighting, and re-running the model
class Trace_SMC(pyro.Messenger):
    def __init__(self, fn=None):
        self.partial_weight = None
        super(Trace_SMC, self).__init__(fn)

    def process_message(self, msg):
        
        # TODO: hack to prevent early termination
        # the replay trace should handle appropriately
        if msg.get("smc_tagged", False):
            msg["value"] = False
            msg["stop"] = False

        # taking a sample which is observed, that's our clue
        # wink wink, nod nod
        if (msg["type"] == "sample" and
            msg["is_observed"] and 
            not msg.get("smc_tagged", False)):

            # HACK: here this tells the computation to stop
            assert type(msg["fn"]) == dist.Bernoulli, \
                "all SMC observe statements must be bernoullis"
            
            # get the score by using bernoulli log prob --
            # also assuming logits were set as score
            self.partial_weight = msg["fn"].log_prob(msg["value"])

            # message will return a 1 
            msg["value"] = True
            
            # in the future replays, we'll ignore these sites and return false
            msg["smc_tagged"] = True

            # we also tell the pyro stack to ignore
            msg["stop"] = True
        
    
    def partial_trace(self, *args, **kwargs):
        # run until we hit an observe statement
        partial_fn_out = self(*args, **kwargs)
        return partial_fn_out, self.partial_weight


def norm_prob(probs):
    return probs / probs.sum()
    
# This is a unified interface for stochastic variational inference in Pyro.
# The actual construction of the loss is taken care of by `loss`.
# See http://docs.pyro.ai/en/0.3.0-release/inference_algos.html
class SMC(object):
    def __init__(self, model, particle_count=100):
        self.model = model
        self.particle_count = particle_count
        
        self.particle_out = None
        self.particle_wgts = None
        self.particle_traces = None


    def _init_particles(self, *args, **kwargs):
        self.particle_out = []
        self.particle_traces = []
        weights = []

        # for every particle 
        for _ in range(self.particle_count):

            # we get a particle trace
            with pyro.trace() as particle_trace:
                # we must run a partial trace for every partial
                partial_out, partial_weight = Trace_SMC(
                    self.model).partial_trace(*args, **kwargs)
                
                # add the sum of the log probs in the partial trace
                # to the weighting
                partial_weight += log_prob_sum(particle_trace)

            # now we have a trace (that we can clone and replace)
            self.particle_out.append(partial_out)
            self.particle_traces.append(particle_trace)
            weights.append(partial_weight)
    
        # set our particle weights 
        self.particle_wgts = torch.Tensor(weights)

    def resample_traces(self):
        # we're going to create a new set outs and traces
        replay_traces = []

        # get our normalized weights
        norm_weights = norm_prob(self.particle_wgts)

        # then let's sample from our categorical (with replace)
        particle_ixs = dist.Categorical(
            probs=norm_weights).sample([self.particle_count])

        # we've got all our particles
        return [self.particle_traces[ix.item()] 
                for ix in particle_ixs]

    def _step_traces(self, particle_replays, *args, **kwargs):
        new_particles = []
        new_traces = []
        new_weights = []
        # now let's replay the traces 
        # for every particle -- then go another partial step
        for i in range(self.particle_count):
            
            # grab our particle replay object
            replay = particle_replays[i]

            # we are going to trace as far as we can
            with pyro.trace() as particle_trace:
                # however this time, we replay the first chunk
                # and then when we get to the next observe
                # we finish!
                with pyro.replay(None, replay):
                    # we must run a partial trace for every partial
                    partial_out, partial_weight = Trace_SMC(
                        self.model).partial_trace(*args, **kwargs)
                    
                    if partial_weight is None: 
                        print("Particle finished generating {}".format(i))
                        # never called a weight, our particle is finished! 
                        partial_weight = 0

                    # add the sum of the log probs in the partial trace
                    # ignoring all the previous replay weights
                    partial_weight += log_prob_sum(
                                        particle_trace,
                                        site_filter=lambda name, site: not name in replay)

            # now we have a trace (that we can clone and replace)
            new_particles.append(partial_out)
            new_traces.append(particle_trace)
            new_weights.append(partial_weight)
        
        weights = torch.Tensor(new_weights)

        self.particle_out = new_particles
        self.particle_traces = new_traces

        # add the weights of the trace
        self.particle_wgts += weights

    # This method handles running the model {particle_count} times,
    # reweighted after observe statements
    def step(self, *args, **kwargs):
        
        # run a single step of smc
        if self.particle_out is None:
            self._init_particles(*args, **kwargs)

            # send back all the current particles
            return self.particle_out, self.particle_wgts

        # resample our particles
        particle_replays = self.resample_traces()

        # we now have our set of new weights
        self._step_traces(particle_replays, *args, **kwargs)

        # everything is set for all our particles
        return self.particle_out, self.particle_wgts


def main(args):

    # let's load our target image to generate vines
    tgt_img = load_target(args.target_img)

    # Generate some data.
    torch.manual_seed(0)

    # Basic training loop
    pyro.get_param_store().clear()

    # TODO: Mini-pyro backend support?
    # Because the API in minipyro matches that of Pyro proper,
    # training code works with generic Pyro implementations.
    # with pyro_backend(args.backend):

    # let's create our SMC object, then do some steps
    smc = SMC(vine_model, particle_count=args.particle_count)

    for step in range(args.num_steps):
        # call with our target_img to match against
        vine_particles, vine_weights = smc.step(tgt_img)
    
    # when we're all done, we display some of our vines
    vine_images = torch.stack(
                    [render_vines(vine, tgt_img.shape[1:], default_viewport)
                     for vine in vine_particles], 0)
    vv.images(vine_images)
    bb()

    print("Finished SMC example")

if __name__ == "__main__":
    # assert pyro.__version__.startswith('0.3.1')
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-b", "--backend", default="minipyro")
    parser.add_argument("-tgt", "--target-img", default="./images/g.png", type=str)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-pc", "--particle-count", default=100, type=int)
    args = parser.parse_args()
    main(args)
