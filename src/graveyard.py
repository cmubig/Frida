

def load_img_internet(url, h=None, w=None):
    response = requests.get(url).content
    im = Image.open(io.BytesIO(response))
    im = np.array(im)
    if im.shape[1] > max_size:
        fact = im.shape[1] / max_size
    im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    im = torch.from_numpy(im)
    im = im.permute(2,0,1)
    #print(im.shape)
    return im.unsqueeze(0).float()


# import clip
# clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

# LPIPS
# loss_fn_alex = lpips.LPIPS(net='alex').to(device)
# lpips_transform = transforms.Compose([transforms.Resize((64,64))])

# import ttools.modules
# perception_loss = ttools.modules.LPIPS().to(device)


def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return augment_trans


# def next_stroke(canvas, target, x_y_attempts=30):
#     with torch.no_grad():
#         diff = torch.mean(torch.abs(canvas[:,:3] - target).unsqueeze(dim=0)**2, dim=2)
#         # diff[diff < 0.2] = 1e-10
#         diff /= diff.sum() # Prob distribution
#         diff = diff.detach().cpu()[0][0].numpy()

#     opt_params = { # Find the optimal parameters
#         'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
#     }

#     for x_y_attempt in range(x_y_attempts):
#         y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)
#         color = target[0,:,y,x]
#         h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]
#         y = torch.from_numpy(np.array(y))/h*2 - 1
#         x = torch.from_numpy(np.array(x))/w*2 - 1

#         for stroke_ind in range(len(strokes_small)):
#             brush_stroke = BrushStroke(stroke_ind, color=color, a=None, xt=x, yt=y).to(device)
#             # opt = torch.optim.Adam(brush_stroke.parameters(), lr=1e-2)
#             # for brush_opt_iter in range(5):
#             #     opt.zero_grad()
#             #     single_stroke = brush_stroke(strokes_small)
#             #     canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
#             #     loss = 0
#             #     loss += nn.L1Loss()(canvas_candidate[:,:3], target)
#             #     loss.backward()
#             #     opt.step()
#             single_stroke = brush_stroke(strokes_small)
#             canvas_candidate = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
#             loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
#             if loss < opt_params['loss']:
#                 opt_params['canvas'] = canvas_candidate
#                 opt_params['loss'] = loss
#                 opt_params['stroke_ind'] = stroke_ind
#                 opt_params['brush_stroke'] = brush_stroke

#     return opt_params['brush_stroke'], opt_params['canvas']


def relax(painting, target, batch_size=20):
    relaxed_brush_strokes = []

    future_canvas_cache = create_canvas_cache(painting)

    #print(painting.background_img.shape)
    canvas_before = T.Resize(size=(strokes_small[0].shape[0],strokes_small[0].shape[1]))(painting.background_img.detach())
    #print(canvas_before.shape)
    for i in tqdm(range(len(painting.brush_strokes))):
        with torch.no_grad():
            canvas_after = torch.zeros((1,4,strokes_small[0].shape[0],strokes_small[0].shape[1])).to(device)
            for j in range(i+1,len(painting.brush_strokes),1):
                brush_stroke = painting.brush_strokes[j]
                single_stroke = brush_stroke(strokes_small)
                if j in future_canvas_cache.keys():
                    canvas_after = canvas_after * (1 - future_canvas_cache[j][:,3:]) + future_canvas_cache[j][:,3:] * future_canvas_cache[j]
                    break
                else:
                    canvas_after = canvas_after * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke

        brush_stroke = painting.brush_strokes[i]
        best_stroke = copy.deepcopy(brush_stroke)
        single_stroke = brush_stroke(strokes_small)
        canvas = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        canvas = canvas * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        best_loss = loss_fcn(canvas, target, use_clip_loss=False)

        for stroke_ind in range(len(strokes_small)):
            brush_stroke.stroke_ind = stroke_ind
            opt = torch.optim.Adam(brush_stroke.parameters(), lr=1e-3)
            for it in range(10):
                opt.zero_grad()
                single_stroke = brush_stroke(strokes_small)
                canvas = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
                canvas = canvas * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after

                loss = loss_fcn(canvas, target, use_clip_loss=False)

                loss.backward()
                opt.step()

                if loss < best_loss:
                    best_stroke = copy.deepcopy(brush_stroke)
                    best_loss = loss

        brush_stroke = best_stroke
        single_stroke = brush_stroke(strokes_small).detach()

        canvas_before = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        relaxed_brush_strokes.append(brush_stroke)

        # global local_it
        # local_it +=1
        # if i % 5 == 0:
        #     relaxed_painting = copy.deepcopy(painting)
        #     relaxed_painting.brush_strokes = nn.ModuleList(relaxed_brush_strokes).extend(painting.brush_strokes[i+1:])
        #     log_progress(relaxed_painting, force_log=True)

    relaxed_painting = copy.deepcopy(painting)
    relaxed_painting.brush_strokes = nn.ModuleList(relaxed_brush_strokes)
    return relaxed_painting



def loss_fcn(painting, target, use_clip_loss=False, use_style_loss=False):
    loss = 0 
    # return loss_l1(painting[:,:3], target)
    #diff = torch.abs(painting[:,:3] - target)
    diff = (painting[:,:3] - target)**2

    # diff[diff>(.1**2)] = diff[diff>(.1**2)] * 10

    # diff = ((torch.nan_to_num(K.color.rgb_to_lab(torch.nan_to_num(painting[:,:3])), nan=0, posinf=127, neginf=-127) \
    #     - torch.nan_to_num(K.color.rgb_to_lab(target), nan=0, posinf=127, neginf=-127))/127.)**2
    
    #diff[diff < 0.2] = 0
    loss += diff.mean()

    # return diff.mean()
    # # diff = torch.abs(normalize_img(painting[:,:3]) - normalize_img(target))

    # #diff = torch.abs(normalize_img(K.color.rgb_to_lab(painting[:,:3])) - normalize_img(K.color.rgb_to_lab(target)))
    # c = K.color.rgb_to_lab(painting[:,:3]) 
    # t = K.color.rgb_to_lab(target)
    # # c -= c.mean()
    # # c /= c.std()
    # # t -= t.mean()
    # # t /= t.std()
    # # c = (c - c.mean()) / c.std()
    # # t = (t - t.mean()) / t.std()
    # diff = torch.abs(c - t) / 127
    # # print(diff.mean())
    # diff[diff<diff.mean()] = 0
    # # diff = diff**2
    # loss += diff.mean()



    if use_clip_loss:
        cl = clip_conv_loss(painting, target) #* 0.5
        opt.writer.add_scalar('loss/content_loss', cl.item(), local_it)
        loss = cl
    if use_style_loss:
        sl = compute_style_loss(painting, target) * .5
        opt.writer.add_scalar('loss/style_loss', sl.item(), local_it)
        loss += sl

        # loss += torch.nn.L1Loss()(K.filters.canny(painting[:,:3])[0], K.filters.canny(target)[0])
    return loss



# def next_stroke_text(canvas, text_features, colors, brush_opt_iters=5):
#     # Random brush stroke
#     brush_stroke = BrushStroke(np.random.randint(len(strokes_small)), color=colors[np.random.randint(len(colors))].clone()).to(device)

#     for p in brush_stroke.parameters(): p.requires_grad = True
#     brush_stroke.color_transform.requires_grad = False

#     opt = torch.optim.Adam(brush_stroke.parameters(), lr=5e-2)
#     for brush_opt_iter in range(brush_opt_iters):
#         opt.zero_grad()
#         single_stroke = brush_stroke(strokes_small)
#         canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
#         loss = clip_text_loss(canvas_candidate, text_features, 4)
#         loss.backward()
#         opt.step()

#     single_stroke = brush_stroke(strokes_small)
#     canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke

#     for p in brush_stroke.parameters(): p.requires_grad = True
#     return brush_stroke, canvas_candidate



def next_stroke_text(canvas, text_features, colors, x_y_attempts=1):
    opt_params = { # Find the optimal parameters
        'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
    }
    for x_y_attempt in range(x_y_attempts):
        # Random brush stroke
        brush_stroke = BrushStroke(np.random.randint(len(strokes_small)), color=colors[np.random.randint(len(colors))].clone()).to(device)
                
        single_stroke = brush_stroke(strokes_small)
        canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        
        with torch.no_grad():
            loss = clip_text_loss(canvas_candidate, text_features, 1).item()

        #loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
        if loss < opt_params['loss']:
            opt_params['canvas'] = canvas_candidate
            opt_params['loss'] = loss
            opt_params['brush_stroke'] = brush_stroke

    return opt_params['brush_stroke'], opt_params['canvas']



            # np_painting = K.filters.canny(p[:,:3])[0].detach().cpu().numpy()[0].transpose(1,2,0)
            # np_painting /= np_painting.max()
            # # print(np_painting.max(), np_painting.min())
            # opt.writer.add_image('images/painting_edges', np_painting*255., j)
            
            # if j== 0:
            #     np_painting = K.filters.canny(target[:,:3])[0].detach().cpu().numpy()[0].transpose(1,2,0)
            #     np_painting /= np_painting.max()
            #     opt.writer.add_image('images/target_edges', np_painting*255., j)

        # # Remove unnecessary brush strokes
        # print("Removing unnecessary brush strokes")
        # n_strokes_before = len(painting.brush_strokes)
        # painting = purge_extraneous_brush_strokes(painting, target)
        # print('Removed {} brush strokes. {} total now.'.format(str(len(painting.brush_strokes) - n_strokes_before), str(len(painting.brush_strokes))))
        




def color_that_can_help_the_most(target, canvas, colors):
    target = cv2.resize(target.copy(), (256,256)) # FOr speed
    canvas = cv2.resize(canvas.copy(), (256,256))

    target = discretize_image(target, colors)
    canvas = discretize_image(canvas, colors)
    # show_img(target/255.)
    # show_img(canvas/255.)

    color_probabilities = []
    for color_ind in range(len(colors)):
        color_pix = target == colors[color_ind][None,None,:]
        color_pix = np.all(color_pix, axis=2)
        #show_img(color_pix)
        #print(color_pix.astype(np.float32).sum(), color_pix.shape)
        # Find how many pixels in the canvas are incorrectly colored
        incorrect_colored_pix = (target[color_pix] != canvas[color_pix]).astype(np.float32).sum()
        color_probabilities.append(incorrect_colored_pix)
    color_probabilities = np.array(color_probabilities)
    #print(color_probabilities)
    color_probabilities /= color_probabilities.sum()

    #print(color_probabilities)
    color_ind = np.random.choice(len(colors), 1, p=color_probabilities)
    return int(color_ind[0])

def paint_coarsely(painter, target, colors):
    # Median filter target image so that it's not detailed
    og_shape = target.shape
    target = cv2.resize(target.copy(), (256,256)) # For scipy memory error
    target = median_filter(target, size=(int(target.shape[0]*0.1),int(target.shape[0]*0.1), 3))
    target = cv2.resize(target, (og_shape[1], og_shape[0]))
    # show_img(target/255., title="Target for the coarse painting phase")
    painter.writer.add_image('target/coarse_target', target/255., 0)

    paint_planner(painter, target, colors,
            how_often_to_get_paint = 4,
            strokes_per_color = 12,
            camera_capture_interval = 12
    )


def paint_finely(painter, target, colors):
    # with torch.no_grad():
    #     target_square = cv2.resize(target.copy(), (256,256))
    #     target_tensor = torch.from_numpy(target_square.transpose(2,0,1)).unsqueeze(0).to(device).float()
    #     #print(target_tensor.shape)
    #     content_mask = get_l2_mask(target_tensor)
    #     #print(content_mask.shape)
    #     content_mask = content_mask.detach().cpu().numpy()[0,0]
    #     content_mask = cv2.resize(content_mask.copy(), (target.shape[1], target.shape[0]))
    #     painter.writer.add_image('target/content_mask', content_mask*255., 0)

    paint_planner(painter, target, colors,
            how_often_to_get_paint = 4,
            strokes_per_color = 8,
            camera_capture_interval = 4
        )
def paint_planner(painter, target, colors,
        how_often_to_get_paint,
        strokes_per_color,
        camera_capture_interval,
        loss_fcn=None):
    camera_capture_interval = 3################

    curr_color = None
    canvas_after = painter.camera.get_canvas()*0. + 255.
    consecutive_paints = 0 
    full_sim_canvas = canvas_after.copy()
    global global_it
    og_target = target.copy()
    real_canvases = [canvas_after]
    sim_canvases = [canvas_after]
    target_lab = rgb2lab(target)


    for it in tqdm(range(2000)):
        canvas_before = canvas_after

        # Decide if and what color paint should be used next
        if it % strokes_per_color == 0:
            # color_ind = int(np.floor(it / 30. )%args.n_colors)
            # Pick which color to work with next based on how much it can help
            # color_ind = 0
            # while color_ind == 0:
            color_ind = color_that_can_help_the_most(target, canvas_before, colors)
            color = colors[color_ind]

        # Plan the next brush stroke
        x, y, stroke_ind, rotation, sim_canvas, loss, diff, stroke_bool_map, int_loss \
            = painter.next_stroke(canvas_before.copy(), target, color, colors, x_y_attempts=6)
        if int_loss > 0:
            # print(loss, 'diddnt help')
            continue # Stroke doesn't actually help

        # Make the brush stroke on a simulated canvas
        full_sim_canvas, _, _ = apply_stroke(full_sim_canvas, painter.strokes[stroke_ind], stroke_ind,
            colors[color_ind], int(x*target.shape[1]), int((1-y)*target.shape[0]), rotation)

        # Clean paint brush and/or get more paint
        new_paint_color = color_ind != curr_color
        if new_paint_color:
            painter.clean_paint_brush()
            curr_color = color_ind
        if consecutive_paints >= how_often_to_get_paint or new_paint_color:
            painter.get_paint(color_ind)
            consecutive_paints = 0

        # Convert the canvas proportion coordinates to meters from robot
        x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

        # Paint the brush stroke
        all_strokes[stroke_ind]().paint(painter, x, y, rotation * (2*3.14/360))

        if it % camera_capture_interval == 0:
            #painter.to_neutral()
            canvas_after = full_sim_canvas#painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
            real_canvases.append(canvas_after)
            sim_canvases.append(full_sim_canvas)

            # Update representation of paint color
            new_paint_color = extract_paint_color(canvas_before, canvas_after, stroke_bool_map)
            color_momentum = 0.15
            if new_paint_color is not None:
                colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                                  + new_paint_color * color_momentum


        if it%5==0: # loggin be sloggin
            painter.writer.add_scalar('loss/loss', 
                np.mean(compare_images(cv2.resize(target_lab, (256,256)), rgb2lab(cv2.resize(canvas_after, (256,256))))), global_it)
            if not painter.opt.simulate:
                painter.writer.add_scalar('loss/sim_loss', 
                    np.mean(compare_images(cv2.resize(target_lab, (256,256)), rgb2lab(cv2.resize(full_sim_canvas, (256,256))))), global_it)
                # painter.writer.add_scalar('loss/sim_stroke', loss, global_it)
                # painter.writer.add_scalar('loss/actual_stroke', np.mean(np.abs(canvas_before - canvas_after)), global_it)
                # painter.writer.add_scalar('loss/sim_actual_diff', \
                #     np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.)), global_it)

        if it % 5 == 0:
            if not painter.opt.simulate:
                painter.writer.add_image('images/canvas', canvas_after/255., global_it)
                # painter.writer.add_image('images/propsed_stroke', sim_canvas, global_it)
                all_colors = save_colors(colors)
                painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
            painter.writer.add_image('images/sim_canvas', full_sim_canvas/255., global_it)

            diff = diff.astype(np.float32) / diff.max() *255. # Normalize for the image
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            diff[:30,:50,:] = color[None, None, :] # Put the paint color in the corner
            painter.writer.add_image('images/diff', diff/255., global_it)
            # painter.writer.add_image('images/sim_actual_diff', 
            #     np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.), axis=2), global_it)
        
            canvas_edges = sobel(np.mean(canvas_after, axis=2))
            canvas_edges -= canvas_edges.min()
            canvas_edges /= canvas_edges.max()
            painter.writer.add_image('images/canvas_edges', canvas_edges*255., global_it)
        if it == 0:
            target_edges = sobel(np.mean(cv2.resize(target, (canvas_after.shape[1], canvas_after.shape[0])), axis=2))
            target_edges -= target_edges.min()
            target_edges /= target_edges.max()
            painter.writer.add_image('target/target_edges', target_edges*255., global_it)
        global_it += 1
        consecutive_paints += 1

        # if it % 20 == 0:
        #     target = discretize_image(og_target, colors)
        #     painter.writer.add_image('target/discrete', target/255., global_it) 

        if it % 100 == 0:
            # to_gif(all_canvases)
            to_video(real_canvases, fn='real_canvases.mp4')
            to_video(sim_canvases, fn='sim_canvases.mp4')




from scipy.signal import medfilt

# from content_loss import get_l2_mask
# import torch
# target_weight = None

target_edges = None

def pick_next_stroke(curr_canvas, target, strokes, color, colors, x_y_attempts, 
        H_coord=None, # Transform the x,y coord so the robot can actually hit the coordinate
        loss_fcn=lambda c,t: np.mean(np.abs(c - t), axis=2)):
    """
    Given the current canvas and target image, pick the next brush stroke
    """
    # It's faster if the images are lower resolution
    fact = target.shape[1] / 196.0 #12.#8.#8.
    curr_canvas = cv2.resize(curr_canvas.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))

    target = cv2.resize(target.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))
    strokes_resized = []
    for stroke in strokes:
        resized_stroke = cv2.resize(stroke.copy(), (int(stroke.shape[1]/fact), int(stroke.shape[0]/fact)))
        strokes_resized.append(resized_stroke)
    strokes = strokes_resized

    # global target_weight 
    # if target_weight is None or target_weight.shape[0] != target.shape[0]:
    #     target_weight = get_l2_mask(target) #+ 0.25

    global target_edges
    if target_edges is None or target_edges.shape[1] != target.shape[1]:
        target_edges = sobel(np.mean(target, axis=2))
        target_edges -= target_edges.min()
        target_edges /= target_edges.max()

    opt_params = { # Find the optimal parameters
        'x':None,
        'y':None,
        'rot':None,
        'stroke':None,
        'canvas':None,
        'loss':9999999,
        'stroke_ind':None,
        'stroke_bool_map':None,
    }

    target_lab = rgb2lab(target)
    curr_canvas_lab = rgb2lab(curr_canvas)


    # Regions of the canvas that the paint color matches well are high
    # color_diff = compare_images(target_lab, rgb2lab(np.ones(target.shape) * color[None,None,:]))
    # color_diff = color_diff.max() - color_diff # Small diff means the paint works well there


    target_discrete = discretize_image(target, colors)
    canvas_discrete = discretize_image(curr_canvas, colors)

    color_pix = np.isclose(target_discrete, np.ones(target.shape) * color[None,None,:])
    color_pix = np.all(color_pix, axis=2).astype(np.float32)
    # Find how many pixels in the canvas are incorrectly colored
    color_diff = np.all(target_discrete != canvas_discrete, axis=2).astype(np.float32)*color_pix \
            + 1e-10
    #show_img(diff)

    loss_og = compare_images(target_lab, rgb2lab(curr_canvas)) 

    # Diff is a combination of canvas areas that are highly different from target
    # and where the paint color could really help the canvas
    diff = (loss_og / loss_og.sum()) * color_diff#(color_diff / color_diff.sum())

    # Ignore edges for now
    t = 0.01
    diff[0:int(diff.shape[0]*t),:] = 0.
    diff[int(diff.shape[0] - diff.shape[0]*t):,:] = 0.
    diff[:,0:int(diff.shape[1]*t)] = 0.
    diff[:,int(diff.shape[1] - diff.shape[1]*t):] = 0.

    # Median filter for smooothness
    # diff = medfilt(diff,kernel_size=5)




    # Turn to probability distribution
    if diff.sum() > 0:
        diff = diff/diff.sum()
    else:
        diff = np.ones(diff.shape, dtype=np.float32) / (diff.shape[0]*diff.shape[1])

    # diff = diff * target_weight
    # diff = diff/diff.sum()

    loss_before = np.sum(compare_images(curr_canvas_lab, target_lab))
    for x_y_attempt in range(x_y_attempts): # Try a few random x/y's
        # Randomly choose x,y locations to start the stroke weighted by the difference in canvas/target wrt color
        y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)
        # y, x = np.random.randint(diff.shape[0]), np.random.randint(diff.shape[1])

        for stroke_ind in range(len(strokes)):
            stroke = strokes[stroke_ind]
            for rot in range(0, 360, 30):

                candidate_canvas, stroke_bool_map, bbox = \
                    apply_stroke(curr_canvas.copy(), stroke, stroke_ind, color, x, y, rot)

                comp_inds = stroke_bool_map > 0.1 # only need to look where the stroke was made
                if comp_inds.sum() == 0:
                    # Likely trying to just paint near the edge of the canvas and rotation took the stroke
                    # off the canvas
                    print('No brush stroke region. Resolution probably too small.')
                    continue

                # Loss is the amount of difference lost
                # loss = -1. * np.sum(diff[comp_inds])

                # Loss is the L1 loss lost
                # loss = np.sum(np.abs(candidate_canvas[comp_inds] - target[comp_inds])) \
                #         - np.sum(np.abs(curr_canvas[comp_inds] - target[comp_inds]))
                #print(target_lab[comp_inds].shape, rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0].shape)
                loss = np.sum(compare_images(rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0], target_lab[comp_inds])) \
                        - np.sum(compare_images(curr_canvas_lab[comp_inds], target_lab[comp_inds]))
                # loss = np.sum(compare_images(rgb2lab(candidate_canvas), target_lab)) \
                #         - loss_before#np.sum(compare_images(curr_canvas_lab, target_lab))
                # loss = np.sum(diff[comp_inds]*(compare_images(rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0], target_lab[comp_inds]) \
                #         - compare_images(curr_canvas_lab[comp_inds], target_lab[comp_inds])))
                
                # loss = np.mean(np.mean(np.abs(candidate_canvas - target), axis=2)* (1. - np.all(target == color, axis=2).astype(np.float32)))

                # Should have the same edges in target and canvas
                # canvas_edges = sobel(np.mean(candidate_canvas, axis=2))
                # canvas_edges -= canvas_edges.min()
                # canvas_edges /= canvas_edges.max()
                # edge_loss_weight = 100.0#1.0e-3
                # # edge_loss = np.mean(np.abs(canvas_edges - target_edges)*target_edges) * edge_loss_weight
                # edge_loss = np.mean(np.abs(canvas_edges - target_edges)) * edge_loss_weight
                # loss += edge_loss

                # Penalize making mistakes (painting in areas it shouldn't)
                # mistakes = compare_images(target_lab[comp_inds], \
                #     rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0])
                # mistake_weight = .5#global_it/50.#1.2
                # loss += (mistakes.sum() / comp_inds.sum()) / 255. * mistake_weight


                if loss < opt_params['loss']:
                    opt_params['x'], opt_params['y'] = x, y
                    opt_params['rot'] = rot
                    opt_params['stroke'] = stroke
                    opt_params['canvas'] = candidate_canvas
                    opt_params['loss'] = loss
                    opt_params['stroke_ind'] = stroke_ind
                    opt_params['stroke_bool_map'] = stroke_bool_map
    if opt_params['loss'] > 0:
        print(opt_params['loss'])
    return 1.*opt_params['x']/curr_canvas.shape[1], 1 - 1.*opt_params['y']/curr_canvas.shape[0],\
            opt_params['stroke_ind'], opt_params['rot'], opt_params['canvas']/255., \
            np.mean(compare_images(target_lab, rgb2lab(opt_params['canvas']))), \
            diff, opt_params['stroke_bool_map'], opt_params['loss']

# binned sorting by color
            # bin_size = 40#100
            # for j in range(0,len(instructions), bin_size):
            #     instructions[j:j+bin_size] = sorted(instructions[j:j+bin_size], key=lambda x : x[5])







    def plan_all_strokes_grid(opt, optim_iter=100, num_strokes_x=30, num_strokes_y=25, 
            x_y_attempts=200, num_passes=1):
    global strokes_small, strokes_full, target
    strokes_small = load_brush_strokes(opt, scale_factor=5)
    strokes_full = load_brush_strokes(opt, scale_factor=1)
    
    # target = load_img(opt.target,
    #     h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.
    target = load_img(os.path.join(opt.cache_dir, 'target_discrete.jpg'),
        h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.

    opt.writer.add_image('target/target', np.clip(target.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), local_it)

    # colors = get_colors(cv2.resize(cv2.imread(opt.target)[:,:,::-1], (256, 256)), n_colors=opt.n_colors)
    with open(os.path.join(opt.cache_dir, 'colors.npy'), 'rb') as f:
        colors = np.load(f)
    colors = (torch.from_numpy(np.array(colors)) / 255.).to(device)


    # Get the background of painting to be the current canvas
    current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg')).to(device)/255.
    current = current_canvas
    painting = Painting(0, background_img=current_canvas, unique_strokes=len(strokes_small)).to(device)

    target_lab = K.color.lab.rgb_to_lab(target)
    for i in range(num_passes):
        canvas = painting(strokes=strokes_small)

        gridded_brush_strokes = []

        h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]

        xys = [(x,y) for x in torch.linspace(-.99,.99,num_strokes_x) for y in torch.linspace(-.99,.99,num_strokes_y)]
        k = 0
        random.shuffle(xys)
        for x,y in tqdm(xys):
            opt_params = { # Find the optimal parameters
                'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
            }
            # solve for stroke type, color, and rotation
            # og_correct_pix = torch.sum(torch.isclose(canvas[:,:3], target, atol=1e-2).float())
            # og_incorrect_pix = torch.sum(1-torch.isclose(canvas[:,:3], target, atol=1e-2).float())
            og_correct_pix = torch.sum(torch.isclose(K.color.lab.rgb_to_lab(canvas[:,:3])/127., target_lab/127., atol=1e-2).float())
            og_incorrect_pix = torch.sum(1-torch.isclose(K.color.lab.rgb_to_lab(canvas[:,:3])/127., target_lab/127., atol=1e-2).float())
            for x_y_attempt in range(x_y_attempts):
                # Random brush stroke
                color = target[:,:3,int((y+1)/2*target.shape[2]), int((x+1)/2*target.shape[3])][0]
                brush_stroke = BrushStroke(np.random.randint(len(strokes_small)), 
                    xt=x,
                    yt=y,
                    a=(np.random.randint(20)-10)/10*3.14,
                    color=color.detach().clone())
                    # color=colors[np.random.randint(len(colors))].clone()).to(device)
                        
                single_stroke = brush_stroke(strokes_small)
                # single_stroke[:,3][single_stroke[:,3] > 0.5] = 1. # opaque
                canvas_candidate = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
                
                with torch.no_grad():
                    #loss = loss_fcn(canvas_candidate, target,  use_clip_loss=False).item()
                    # diff = torch.abs(canvas_candidate - target)
                    # # diff[diff>0.1] = 10.
                    # loss = diff.mean()

                    # correct_pix = torch.sum(torch.isclose(canvas_candidate[:,:3], target, atol=1e-2).float())
                    # incorrect_pix = torch.sum(1-torch.isclose(canvas_candidate[:,:3], target, atol=1e-2).float())
                    # loss = -1. * (correct_pix - og_correct_pix)
                    # loss += (incorrect_pix - og_incorrect_pix)

                    canvas_candidate_lab = K.color.lab.rgb_to_lab(canvas_candidate)
                    # loss = torch.mean((canvas_candidate_lab - target_lab)**2)

                    correct_pix = torch.sum(torch.isclose(canvas_candidate_lab[:,:3]/127., target_lab/127., atol=1e-2).float())
                    incorrect_pix = torch.sum(1-torch.isclose(canvas_candidate_lab[:,:3]/127., target_lab/127., atol=1e-2).float())
                    loss = -1. * (correct_pix - og_correct_pix)
                    loss += (incorrect_pix - og_incorrect_pix) * 4

                    # chg_inds = (single_stroke[0,-1] > 0.2)

                    # loss_acc = torch.abs(canvas_candidate[:,:3,chg_inds] - target[:,:,chg_inds]).mean() \
                    #         - torch.abs(canvas[:,:3,chg_inds] - target[:,:,chg_inds]).mean()
                    # loss_big = torch.abs(canvas_candidate[:,:3,chg_inds] - target[:,:,chg_inds]).sum() \
                    #         - torch.abs(canvas[:,:3,chg_inds] - target[:,:,chg_inds]).sum()

                    # # print(loss_big.item(), loss_acc.item())
                    # loss = (1-(k/len(xys))) * (loss_big/1000) + (k/len(xys)) * loss_acc

                    # loss += loss_acc
                    # loss = (1-(k/len(xys))) * (loss) + (k/len(xys)) * loss_acc
                #loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
                if loss < opt_params['loss']:
                    opt_params['canvas'] = canvas_candidate
                    opt_params['loss'] = loss
                    opt_params['brush_stroke'] = brush_stroke
            canvas = opt_params['canvas']
            gridded_brush_strokes.append(opt_params['brush_stroke'])

            # Do a lil optimization on the most recent strokes
            if k % 30 == 0:
                strokes_to_optimize = gridded_brush_strokes[-100:]
                older_strokes = gridded_brush_strokes[:-100]
                back_p = Painting(0, background_img=current_canvas, 
                    brush_strokes=[bs for bs in painting.brush_strokes] + older_strokes).to(device)
                background_img = back_p(strokes=strokes_small, use_alpha=False)
                p = Painting(0, background_img=background_img, 
                    brush_strokes=[bs for bs in painting.brush_strokes] + strokes_to_optimize).to(device)
                optim = torch.optim.Adam(p.parameters(), lr=1e-2)# * (len(painting.brush_strokes)/100))
                for j in (range(10)):
                    optim.zero_grad()
                    loss = loss_fcn(p(strokes=strokes_small, use_alpha=False), target,  use_clip_loss=False, use_style_loss=False)
                    loss.backward()
                    for bs in p.brush_strokes:
                        bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
                    optim.step()

                with torch.no_grad():
                    gridded_brush_strokes = older_strokes
                    gridded_brush_strokes += [bs for bs in p.brush_strokes]
                    p = Painting(0, background_img=current_canvas, 
                        brush_strokes=[bs for bs in painting.brush_strokes] + gridded_brush_strokes).to(device)
                

                # p = Painting(0, background_img=current_canvas, 
                #     brush_strokes=gridded_brush_strokes).to(device)
                # optim = torch.optim.Adam(p.parameters(), lr=1e-2)# * (len(painting.brush_strokes)/100))
                # for j in (range(10)):
                #     optim.zero_grad()
                #     loss = loss_fcn(p(strokes=strokes_small, use_alpha=False), target,  use_clip_loss=False, use_style_loss=False)
                #     # loss = loss_fcn(p(strokes=strokes_small, use_alpha=False), target,  use_clip_loss=True, use_style_loss=False)
                #     loss.backward()
                #     # for bs in p.brush_strokes:
                #     #     bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
                #     optim.step()

                # with torch.no_grad():
                #     gridded_brush_strokes = [bs for bs in p.brush_strokes]
                #     p = Painting(0, background_img=current_canvas, 
                #         brush_strokes=[bs for bs in painting.brush_strokes] + gridded_brush_strokes).to(device)
                
                discretize_colors(p, colors)
                p = sort_brush_strokes_by_color(p)
                # n_strokes = len(p.brush_strokes)
                # p = purge_buried_brush_strokes(p)
                # if len(p.brush_strokes) != n_strokes:
                #     print('removed', n_strokes - len(p.brush_strokes), 'brush strokes')
                n_strokes = len(p.brush_strokes)
                p = purge_extraneous_brush_strokes(p, target)
                if len(p.brush_strokes) != n_strokes:
                    print('removed', n_strokes - len(p.brush_strokes), 'brush strokes that did not help')
                
                with torch.no_grad():
                    gridded_brush_strokes = [bs for bs in p.brush_strokes]
                    canvas = p(strokes=strokes_small, use_alpha=False)

            # painting = Painting(0, background_img=current_canvas, brush_strokes=gridded_brush_strokes).to(device)
            # log_progress(painting)
            if k % 20 == 0:
                np_painting = canvas.detach().cpu().numpy()[0].transpose(1,2,0)
                opt.writer.add_image('images/grid_add', np.clip(np_painting, a_min=0, a_max=1), k)
            k += 1
        painting = Painting(0, background_img=current_canvas, 
            brush_strokes=[bs for bs in painting.brush_strokes] + gridded_brush_strokes).to(device)
        discretize_colors(painting, colors)
        log_progress(painting)

        
        # Optimize all brush strokes
        print('Optimizing all {} brush strokes'.format(str(len(painting.brush_strokes))))
        optim = torch.optim.Adam(painting.parameters(), lr=1e-2)# * (len(painting.brush_strokes)/100))
        for j in tqdm(range(optim_iter)):
            optim.zero_grad()
            p = painting(strokes=strokes_small, use_alpha=False)
            loss = 0
            loss += loss_fcn(p, target,  use_clip_loss=True, use_style_loss=False)
            loss.backward()
            if True:#j > .85*optim_iter: # Only change colors in the beginning of opt
                for bs in painting.brush_strokes:
                    bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
            optim.step()
            log_progress(painting)

            optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.99

            if j % 10 == 0 and (j > .25*optim_iter):
                discretize_colors(painting, colors)
                painting = sort_brush_strokes_by_color(painting)
                optim = torch.optim.Adam(painting.parameters(), lr=optim.param_groups[0]['lr'])

            
    return painting