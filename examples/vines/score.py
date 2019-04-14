from ipdb import set_trace as bb
from render import (
    render_vines, 
    make_gradient_weight_similarity_fct,
    normalized_similarity
    )

import visdom
vv = visdom.Visdom(env="vines")


# Basically Gaussian log-likelihood, without the constant factor
def makescore(val, target, tightness):
    diff = val - target
    return - (diff**2) / (tightness**2)

default_viewport = {"xmin": -12, "xmax": 12, "ymin": -22, "ymax": 2}

def score_vine(
        vines, 
        target_img, 
        sim_tightness=.02,
        # bounds_tightness=.001,
        viewport = default_viewport,
        visualize=False):
    
    # get our shape from the target info
    width, height = target_img.shape[1:]

    # let's get our render
    img = render_vines(vines, [width, height], viewport)
    
    if visualize:
        vv.image(img)

    # now let's get our scoring
    sim_fct = make_gradient_weight_similarity_fct(1.5)

    # now we're ready to score
    
    # Similarity factor
    norm_sim_val = normalized_similarity(img, target_img, sim_fct)

    # then we make our scores
    # similar to gaussian llk
    render_score = makescore(norm_sim_val, 1, sim_tightness)

    # TODO: bounding box comparisons 
    # where we check how well the bounding box of the generated
    # object matched the viewport
    
    # for now, return render_score, later maybe multiple scoring
    return render_score, img

