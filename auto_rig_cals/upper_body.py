
import os
import sys
import json
import numpy as np
import trimesh
import math
def triangulate_face(face):
    """
    face: 예) [0,1,2,3] 처럼 인덱스들의 리스트 (버텍스 인덱스)
    return: 삼각형 인덱스 리스트들의 집합 (길이가 3인 리스트를 여러개)
    """
    if len(face) < 3:
        # 2개 이하라면 삼각형 만들 수 없음
        return []
    elif len(face) == 3:
        # 이미 삼각형
        return [face]
    else:
        # n-gon 이면, (face[0], face[i], face[i+1]) fan triangulation
        triangles = []
        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i+1]])
        return triangles
def vectorize3(lst):
    """주어진 리스트를 float64 numpy 배열로 변환"""
    return np.array(lst, dtype=np.float64)
def tolerance_check_2(source, target, axis, axis2, tolerance, side):
    if source[axis] <= target[axis] + tolerance and source[axis] >= target[axis] - tolerance:
        if source[axis2] <= target[axis2] + tolerance and source[axis2] >= target[axis2] - tolerance:
            #one side only
            if side == ".l":
                if source[0] > 0:
                    return True
            if side == ".r":
                if source[0] < 0:
                    return True
def raycast_first_hit(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray, max_dist: float = None):
    """
    Blender BVHTree의 ray_cast와 유사하게,
    - mesh: trimesh.Trimesh
    - origin, direction: (3,) shape numpy array
    - max_dist: (float or None). None이면 무제한
    반환값: (hit_location, normal, face_idx, distance)
        hit_location: np.array([x,y,z]) or None
        normal      : np.array([nx,ny,nz]) or None
        face_idx    : int (충돌한 face 인덱스) or None
        distance    : float or None
    """
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    origins = origin.reshape(1, 3) 
    directions = direction.reshape(1, 3)
    # multiple_hits=True 로 한 번에 모든 교차점 찾기
    # (BVHTree.ray_cast가 한 번만 부르면 가장 가까운 1개를 돌려주는 것과 유사하게 하려면
    #  여기서 우리가 "가장 가까운 1개"를 선택해야 함)
    locations, index_ray, index_tri = intersector.intersects_location(
        origins, directions, multiple_hits=True
    )
    if len(locations) == 0:
        return (None, None, None, None)

    # 여러 교차점 중 "origin에서 가장 가까운" 점만 고르기
    dist_arr = np.linalg.norm(locations - origin, axis=1)
    min_idx = np.argmin(dist_arr)
    closest_loc = locations[min_idx]
    closest_dist = dist_arr[min_idx]
    face_idx = index_tri[min_idx]

    if max_dist is not None and closest_dist > max_dist:
        return (None, None, None, None)

    normal = mesh.face_normals[face_idx] if face_idx is not None else None
    return (closest_loc, normal, face_idx, closest_dist)
    
def project_point_onto_line(a, b, p):
    # project the point p onto the line a,b
    ap = p - a
    ab = b - a
    result_pos = a + ap.dot(ab) / ab.dot(ab) * ab
    return result_pos

def raycast_front_and_back(mesh, origin, direction, max_dist, min_hit_dist=0.001):
    """
    Mimics your repeated “front -> back” logic by:
      1) Casting from 'origin' in 'direction' up to 'max_dist' 
      2) If a hit is found:
          - front = hit[1]-coordinate
          - Then keep stepping forward (with +small epsilon along the ray)
            until no further hits are found => back
      3) Return (front, back).

    If no hit is found, returns (None, None).
    """
    # 1) Find the first hit
    hit_loc, _, _, distance = raycast_first_hit(mesh, origin, direction, max_dist=max_dist)
    if hit_loc is None or (distance is not None and distance < min_hit_dist):
        return (None, None)

    # 'front' is the y-value of the first hit
    front = hit_loc[1]

    # Now we do a loop, step from that last_hit forward by an offset (0.001 along your code’s normal axis).
    # We keep track of the final y as 'back'.
    have_hit  = True
    last_hit  = hit_loc
    step_vec  = vectorize3([0, 0.001, 0])  # your code uses (0, 0.001, 0)

    while have_hit:
        have_hit = False
        # The new origin is last_hit + small offset
        new_origin = last_hit + step_vec
        hit_loc2, _, _, distance2 = raycast_first_hit(mesh, new_origin, direction, max_dist=max_dist)
        if hit_loc2 is not None:
            have_hit = True
            last_hit = hit_loc2

    back = last_hit[1]
    return (front, back)

def part1 (
        mesh : trimesh.Trimesh,
        hand_marker_location : np.ndarray, 
        body_depth : float, 
        side_idx : int, 
        is_debug : bool, 
        shoulder_location : np.ndarray, 
        shoulder_pos_location : np.ndarray , 
        body_width : float):
    

    if side_idx == 0:
        print('\n[Left arm detection...]')
    if side_idx == 1:
        print('\n[Right arm detection...]')
    suff = ""
    side = ".l"
    if side_idx == 1:
        suff = "_sym"
        side = ".r"

    ray_origin = hand_marker_location + vectorize3([0, -body_depth*5, 0])
    ray_dir = vectorize3([0, body_depth*50, 0])

    (wrist_bound_back, wrist_bound_front) = raycast_front_and_back(mesh, ray_origin, ray_dir, max_dist=np.linalg.norm(ray_dir))
    # delete the json file

    if wrist_bound_back == None:
        error_message = 'Could not find the wrist, marker out of mesh?'
        error_during_auto_detect = True
    hand_loc_x = hand_marker_location[0]
    hand_loc_y = wrist_bound_back + ((wrist_bound_front - wrist_bound_back)*0.4)
    hand_loc_z = hand_marker_location[2]

    if side_idx == 0:
        hand_empty_loc_l = [hand_loc_x, hand_loc_y, hand_loc_z]
    if side_idx == 1:
        hand_empty_loc_r = [hand_loc_x, hand_loc_y, hand_loc_z]



    # ARMS -------

    print("    Find arms...\n")
    shoulder_front = None
    shoulder_back = None

    if is_debug:
        print("    Find shoulders...\n")


    ray_origin = shoulder_location + vectorize3([0, -body_depth*2, 0])
    ray_dir = vectorize3([0, body_depth*4, 0])

    (shoulder_back, shoulder_front) = raycast_front_and_back(mesh, ray_origin, ray_dir, max_dist= np.linalg.norm(ray_dir))
     
    shoulder_empty_loc = [shoulder_location[0], shoulder_back + (shoulder_front-shoulder_back)*0.4, shoulder_location[2]]

    # Shoulder_base
    # Y position: best to bring it forward for best compatibility with humanoid rigs (unreal)
    shoulder_base_loc = [shoulder_empty_loc[0]/4, shoulder_back + (shoulder_front-shoulder_back)*0.8, shoulder_empty_loc[2]]


    # Elbow
    _hand_empty_loc = hand_empty_loc_l
    if side_idx == 1:
        _hand_empty_loc = hand_empty_loc_r

    elbow_empty_loc = [(shoulder_empty_loc[0] + _hand_empty_loc[0])/2, 0, (shoulder_empty_loc[2] + _hand_empty_loc[2])/2]
    elbow_empty_loc = vectorize3(elbow_empty_loc)

    # Find the elbow boundaries

    if is_debug:
        print("    Find elbow boundaries...\n")

    # Get the arm X angle
    #   opposite angle for the right side
    fac = 1
    if side_idx == 1:
        fac = -1

    hand_pos_plane_x = vectorize3(hand_marker_location)  # np.array로 변환
    hand_pos_plane_x[1] = 0.0

    shoulder_pos_plane_x = vectorize3(shoulder_pos_location)  # np.array로 변환
    shoulder_pos_plane_x[1] = 0.0

    # 이제 두 numpy 배열끼리 뺄셈이 가능하므로, 코사인 값을 계산할 수 있습니다.
    arm_angle_x = np.arccos(np.clip(
        np.dot(hand_pos_plane_x - shoulder_pos_plane_x, [fac, 0.0, 0.0]) /
        (np.linalg.norm(hand_pos_plane_x - shoulder_pos_plane_x) * np.linalg.norm([fac, 0.0, 0.0])),
        -1.0, 1.0))
    angle = -arm_angle_x * fac  
    mat_angle_x = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0,             1, 0,             0],
        [-np.sin(angle),0, np.cos(angle), 0],
        [0,             0, 0,             1]
    ])


    # evaluate nearby verts
    # clear_selection()
    elbow_selection = []
    has_selected_v = False
    sel_rad = body_width / 20

    while not has_selected_v:
        for i, v in enumerate(mesh.vertices):
            # v는 이미 [x, y, z]인 numpy 배열이므로 바로 사용합니다.
            if tolerance_check_2(v, elbow_empty_loc, 0, 2, sel_rad, side):
                has_selected_v = True
                elbow_selection.append(i)
        if not has_selected_v:
            sel_rad *= 2

    #if side_idx == 1:
    #    print(br)

    elbow_back = -1000
    elbow_front = 1000
    vert_up = None
    vert_low = None
    elbow_up = -10000
    elbow_low = 10000
    
    for v_idx in elbow_selection:
        vert_y = mesh.vertices[v_idx][1]
        if vert_y < elbow_front:
            elbow_front = vert_y
        if vert_y > elbow_back:
            elbow_back = vert_y

        # 동차 좌표로 변환 후 회전 행렬과 곱하기
        vert_h = np.append(mesh.vertices[v_idx], 1)
        vert_z = (mat_angle_x @ vert_h)[2]
        if vert_up is None:
            vert_up = v_idx
        if vert_low is None:
            vert_low = v_idx
        if vert_z > elbow_up:
            elbow_up = vert_z
            vert_up = v_idx
        if vert_z < elbow_low:
            elbow_low = vert_z
            vert_low = v_idx

    # adust elbow height, in arm space, to better fit the elbow position
    # get middle elbow point
    p = (mesh.vertices[vert_up] + mesh.vertices[vert_low]) * 0.5
    p[1] = 0.0
    # get arm line points
    line_a = vectorize3(shoulder_empty_loc.copy())
    line_a[1] = 0.0
    line_b = vectorize3(_hand_empty_loc.copy())
    line_b[1] = 0.0
    # project middle point onto the arm line
    p_proj = project_point_onto_line(line_a, line_b, p)

    # <!> only apply the evaluated elbow height, if the elbow angle exceeds a given threshold
    # because straight arms are always better for IK chains vector
    elbow_angle = math.degrees(np.arccos(np.clip(
        np.dot(p-line_a, line_b-line_a) /
        (np.linalg.norm(p-line_a) * np.linalg.norm(line_b-line_a)),
        -1.0, 1.0)))
    print("Elbow Angle:", elbow_angle)

    if elbow_angle > 3.6:
        # get the resulting vector
        vec = p-p_proj
        elbow_empty_loc += vec

    # adjust elbow depth
    elbow_empty_loc[1] = elbow_back + (elbow_front - elbow_back)*0.3

    elbow_center = elbow_empty_loc.copy()
    elbow_center[1]  = elbow_back + (elbow_front - elbow_back)*0.5
    
    _hand_empty_loc = hand_empty_loc_l
    if side_idx == 1:
        _hand_empty_loc = hand_empty_loc_r
    return {
        "shoulder_empty_loc": shoulder_empty_loc.tolist() if isinstance(shoulder_empty_loc, np.ndarray) else shoulder_empty_loc,
        "shoulder_base_loc": shoulder_base_loc.tolist() if isinstance(shoulder_base_loc, np.ndarray) else shoulder_base_loc,
        "elbow_empty_loc": elbow_empty_loc.tolist() if isinstance(elbow_empty_loc, np.ndarray) else elbow_empty_loc,
        "elbow_center": elbow_center.tolist() if isinstance(elbow_center, np.ndarray) else elbow_center,
        "elbow_angle": float(elbow_angle),
        "wrist_bound_back": float(wrist_bound_back) if wrist_bound_back is not None else None,
        "wrist_bound_front": float(wrist_bound_front) if wrist_bound_front is not None else None,
        "hand_empty_loc": _hand_empty_loc.tolist() if isinstance(_hand_empty_loc, np.ndarray) else _hand_empty_loc,
        "arm_angle_x": float(arm_angle_x),
    }

