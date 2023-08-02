import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
# from rs_imu import *
from YOLO_Detection import *

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

def monitoring():
    global posisi_end_effector
    state = AppState()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color,640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    state.decimate = (state.decimate + 2) % 3
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()


    def mouse_cb(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)


    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(state.WIN_NAME, window_width, window_height)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


    def project(v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def view(v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


    def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = project(pt1.reshape(-1, 3))[0]
        p1 = project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
                view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
                view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height
        # for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], 1.08)
            # print(p)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        # print(view(top_left))
        # print(view(top_right))
        # print(view(bottom_right))
        # print(view(bottom_left))
        
        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


    def pointcloud(out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]


    out = np.empty((h, w, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(1)

    # Private variable
    # titik_utama = np.array([0.015,0.19,0])
    
    list_detect = None
    i = 0
    j = 0
    edge_wound = []
    wound_coor_depth = []
    segment = []
    color_segment = []
    while True:
        titik_utama = np.array([posisi_end_effector[0]+15, 190-posisi_end_effector[1], 1080-posisi_end_effector[2]])/1000
        # Render
        now = time.time()
        
        # Grab camera data
        if not state.paused:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            success, cap_frame = cap.read()
            cap_frame = cv2.rotate(cap_frame, cv2.ROTATE_180)

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
        
            if not depth_frame or not color_frame:
                continue
            decimate_frame = decimate.process(depth_frame)
        
            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                decimate_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(decimate_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image_height, color_image_width = color_image.shape[:2]
            ROI_image = np.array([[(240,350),(240, 120),(437,120), (437, 350)]], dtype= np.int32)
            blank_image= np.zeros_like(color_image)
            ROI_color_image = cv2.fillPoly(blank_image, ROI_image,(255,255,255))
            color_image = cv2.bitwise_and(color_image, ROI_color_image)
            calc_depth = decimate_frame.as_depth_frame()
            
            if i%2 == 1 :
                if not segment :
                    color_image, list_detect, segment_edge, segment= YOLODetect(color_image)
                    for r in range(len(list_detect)):
                        if list_detect[r][2] == 1 : colorize = (0,0,255)
                        elif list_detect[r][2] == 0 : colorize = (255,0,0)
                        color_segment.append([colorize])
                for segmentation in range(len(list_detect)):

                    # if list_detect[segmentation][2] == 1:
                    h_wound_depth,w_wound_depth = (list_detect[segmentation][1]).astype(int), (list_detect[segmentation][0]).astype(int)
                    z_wound = calc_depth.get_distance(w_wound_depth,h_wound_depth) - 0.01
                    # print(w_wound_depth,h_wound_depth)
                    h_wound_depth,w_wound_depth =  ((355-h_wound_depth)*(0.813/480)), ((w_wound_depth-338)*(1.15/640))
                    # points = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [w_wound_depth, h_wound_depth], z_wound)
                    # points = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [w_wound_depth, h_wound_depth], z_wound)
                    # w_wound_depth = points[0]
                    # h_wound_depth = points[1]
                    wound_coor_depth = []
                    wound_coor_depth.append([w_wound_depth, h_wound_depth, 1.08-z_wound])
                    # wound_coor_depth.append([w_wound_depth - 0.0145, h_wound_depth+0.295, 1.05-z_wound])
                    # wound_coor_depth.append([w_wound_depth, h_wound_depth, 1.05-z_wound])
                    
                    # print(wound_coor_depth)
                # print(point_value)
                # print(points)
                for value in range(len(segment_edge)):
                    x_edge_min = (segment_edge[value][0]-338)*(1.15/640)
                    x_edge_max = (segment_edge[value][2]-338)*(1.15/640)
                    y_edge_min = (355-segment_edge[value][1])*(0.813/480)
                    y_edge_max = (355-segment_edge[value][3])*(0.813/480)
                    z_edge_min = calc_depth.get_distance(segment_edge[value][0],segment_edge[value][1])-0.01
                    z_edge_max = calc_depth.get_distance(segment_edge[value][2],segment_edge[value][3])-0.01
                    if len(edge_wound) > len(wound_coor_depth) : edge_wound = edge_wound[1:]
                    edge_wound.append([x_edge_min,y_edge_min,z_edge_min,x_edge_max,y_edge_max,z_edge_max])

                for inc in range(len(edge_wound)):
                    print(f"Posisi Ujung Wound {inc} Ujung Pertama :", edge_wound[0],edge_wound[1],edge_wound[2])
                    print(f"Posisi Ujung Wound {inc} Ujung Kedua :", edge_wound[3],edge_wound[4],edge_wound[5])
                
                for f in range(len(segment)):
                    points = segment[f].reshape((-1, 1, 2))
                    overlay = img.copy()
                    # print(point_segment)
                    cv2.fillPoly(img, [points], color_segment[f] )
                    img = cv2.addWeighted(overlay, 0.5, img, 1, 0)
            else :
                color_image,list_detect = color_image,list_detect
            
            
            depth_colormap = np.asanyarray(
                colorizer.colorize(decimate_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image

            else:
                mapped_frame, color_source = decimate_frame, depth_colormap
            points = pc.calculate(decimate_frame)
            pc.map_to(mapped_frame)
            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        out.fill(0)
        # grid(out, (0, 0.5, 1), size=1, n=10)
        # cv2.rectangle(color_image, (240, 120), (437, 350), (255, 255, 255), thickness=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)
        for ax in range(len(edge_wound)):
            new_edge_wound_bottom = np.array([edge_wound[ax][0]+0.015,0.195-edge_wound[ax][1],edge_wound[ax][2]])
            new_edge_wound_upper = np.array([edge_wound[ax][3]+0.015,0.195-edge_wound[ax][4],edge_wound[ax][5]])
            line3d(out, view(titik_utama), view(new_edge_wound_bottom), (0, 0xff, 0), 1)
            line3d(out, view(titik_utama), view(new_edge_wound_upper),(0, 0xff, 0), 1)
        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now
        if dt == 0 : dt = 0.0001
        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
        horizontal_cap =np.concatenate((out,cap_frame),axis=1)
        cv2.imshow(state.WIN_NAME, horizontal_cap)
        key = cv2.waitKey(1)

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            i += 1
        #     state.decimate = (state.decimate + 1) % 3
        #     decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("s"):
            j += 1
            cv2.imwrite(f"./out_cloud_{j}.png", out)

        if key == ord("e"):
            points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

    # Stop streaming
    pipeline.stop()
    cap.release()
    cv2.destroyAllWindows()