import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import os
import threading
from ultralytics import YOLO
import math
from concurrent.futures import ThreadPoolExecutor
import csv
import shutil

# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Mode configuration ──────────────────────────────────────────────────────
# Each mode defines the non-red stain class and the UI colour scheme.
# 'cls'          → internal class key stored in every detection box
# 'cls_label'    → human-readable name shown in the UI
# 'cls_key'      → keyboard shortcut to select this class
# 'weights'      → YOLO weights filename
# 'canvas_color' → tkinter colour name for drawing on the canvas
# 'bgr_color'    → OpenCV BGR colour for saving annotated images
# 'primary', 'accent', 'success' → UI colour scheme

MODE_CONFIGS = {
    'green': {
        'cls': 'G',
        'cls_label': 'Green',
        'cls_key': 'g',
        'weights': 'G.pt',
        'canvas_color': 'green',
        'bgr_color': (0, 255, 0),
        'primary': '#2563eb',
        'accent': '#3b82f6',
        'success': '#22c55e',
    },
    'blue': {
        'cls': 'B',
        'cls_label': 'Blue',
        'cls_key': 'b',
        'weights': 'B.pt',
        'canvas_color': 'blue',
        'bgr_color': (255, 0, 0),
        'primary': '#22c55e',
        'accent': '#4ade80',
        'success': '#2563eb',
    },
}

# These globals are set once after the user picks a mode in the splash dialog.
MODE = None          # one of the dicts above
COLORS = None        # full colour palette (depends on mode)
CLASS_COLORS = None  # {cls: bgr, 'R': bgr}


def apply_mode(mode_key):
    """Set the global MODE, COLORS and CLASS_COLORS from a mode key."""
    global MODE, COLORS, CLASS_COLORS
    MODE = MODE_CONFIGS[mode_key]
    COLORS = {
        'primary': MODE['primary'],
        'secondary': '#4b5563',
        'accent': MODE['accent'],
        'success': MODE['success'],
        'warning': '#f59e0b',
        'error': '#ef4444',
        'background': '#ffffff',
        'surface': '#f3f4f6',
        'text': '#1f2937',
    }
    CLASS_COLORS = {
        MODE['cls']: MODE['bgr_color'],
        'R': (0, 0, 255),
    }


def choose_mode():
    """Show a splash dialog and return the chosen mode key ('green' or 'blue')."""
    chosen = {'value': None}

    dialog = ctk.CTk()
    dialog.title("SBeeVia")
    dialog.geometry("420x260")
    dialog.resizable(False, False)

    # Centre on screen
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() - 420) // 2
    y = (dialog.winfo_screenheight() - 260) // 2
    dialog.geometry(f"+{x}+{y}")

    ctk.CTkLabel(
        dialog, text="Sperm Viability Analysis",
        font=("Inter", 22, "bold"), text_color="#1f2937"
    ).pack(pady=(30, 5))

    ctk.CTkLabel(
        dialog, text="Select the colour combination you want to detect:",
        font=("Inter", 13), text_color="#4b5563"
    ).pack(pady=(0, 20))

    btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
    btn_frame.pack(pady=(0, 10))

    def pick(key):
        chosen['value'] = key
        dialog.destroy()

    ctk.CTkButton(
        btn_frame, text="Green/Red",
        width=160, height=50,
        font=("Inter", 15, "bold"),
        fg_color="#22c55e", hover_color="#1ba350",
        command=lambda: pick('green')
    ).pack(side="left", padx=10)

    ctk.CTkButton(
        btn_frame, text="Blue/Red",
        width=160, height=50,
        font=("Inter", 15, "bold"),
        fg_color="#2563eb", hover_color="#1d4ed8",
        command=lambda: pick('blue')
    ).pack(side="left", padx=10)

    ctk.CTkLabel(
        dialog, text="This can be changed later from the sidebar",
        font=("Inter", 13), text_color="#4b5563"
    ).pack(pady=(5, 0))

    dialog.protocol("WM_DELETE_WINDOW", lambda: (pick(None)))
    dialog.mainloop()

    return chosen['value']

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_resource_path(relative_path):
    """Gets the absolute path of a resource, whether in development or executable."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_image_safely(image_path):
    """Load image using cv2 and convert to PIL Image to avoid truncation errors"""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ── Image viewer ─────────────────────────────────────────────────────────────

class ModernTiledImageViewer(ctk.CTkFrame):
    def __init__(self, parent, GUI):
        super().__init__(parent)
        self.parent = parent
        self.GUI = GUI

        # Storage for all boxes with their confidence scores and class
        self.all_boxes = []  # [x1, y1, x2, y2, confidence, class_id]
        self.confidence_thresholds = {MODE['cls']: 0.1, 'R': 0.1}

        # Add ROI variables
        self.roi_points = []
        self.roi_polygons = {}
        self.drawing_roi = False
        self.current_roi_line = None
        self.hover_roi = False

        # Create a canvas with modern styling
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"][1],
            cursor="arrow"
        )

        # Modern scrollbars
        self.v_scroll = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set
        )

        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Initialize variables
        self.original_image = None
        self.image_path = None
        self.tile_cache = {}
        self.tile_size = 1024
        self.scale = 1.0
        self.current_box = None
        self.drag_start = None
        self.initial_scale = None
        self.selected_class = MODE['cls']

        # Box interaction variables
        self.active_box = None
        self.resize_handle = None
        self.box_drag_mode = None
        self.hover_box = None
        self.corner_radius = 5
        self.boxes_hidden = False
        self.hide_key_pressed = False
        self.edge_sensitivity = 5
        self.is_drawing_new = False

        # Bind keyboard events to root window
        self.root = self.winfo_toplevel()
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

        # Create loading label
        self.loading_label = ctk.CTkLabel(self, text="Loading image...")

        # Status message on canvas
        self._status_text_id = None

        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Bind events
        self._bind_events()

        # Re-centre the status text when the canvas is resized
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Show initial guidance
        self.show_status_message("Select a folder to begin the analysis")

    # ── ROI ───────────────────────────────────────────────────────────────

    def show_status_message(self, text):
        """Display a centred guidance message on the canvas."""
        self.clear_status_message()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = 800
        if h <= 1:
            h = 600
        self._status_text_id = self.canvas.create_text(
            w / 2, h / 2,
            text=text,
            font=("Inter", 16),
            fill="#9ca3af",
            tags="status_msg",
        )

    def clear_status_message(self):
        """Remove any guidance message from the canvas."""
        self.canvas.delete("status_msg")
        self._status_text_id = None

    def _on_canvas_configure(self, event):
        """Re-centre the status text when the canvas resizes."""
        if self._status_text_id:
            self.canvas.coords(self._status_text_id, event.width / 2, event.height / 2)

    def start_roi_drawing(self):
        if not self.GUI.current_image:
            return
        self.drawing_roi = True
        self.roi_points = []
        self.canvas.config(cursor="cross")
        self.canvas.bind("<Button-1>", self.add_roi_point)
        self.canvas.bind("<Motion>", self.update_roi_preview)
        self.canvas.bind("<Button-3>", self.delete_roi)
        self.canvas.bind("<Double-Button-1>", self.complete_roi)

    def delete_roi(self, event=None):
        if self.GUI.current_image in self.roi_polygons:
            del self.roi_polygons[self.GUI.current_image]
            self.canvas.delete("roi_line", "roi_point", "roi_preview")
            self.GUI.update_box_statistics()
            self.GUI.roi_button.configure(
                text="Edit ROI",
                fg_color=COLORS['secondary']
            )
            if self.drawing_roi:
                self.stop_roi_drawing()

    def stop_roi_drawing(self):
        self.drawing_roi = False
        self.canvas.config(cursor="arrow")
        self.roi_points = []
        if self.current_roi_line:
            self.canvas.delete(self.current_roi_line)
            self.current_roi_line = None
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-3>", self.delete_box)

    def add_roi_point(self, event):
        if not self.drawing_roi:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.roi_points.append((canvas_x, canvas_y))
        point_radius = 3
        self.canvas.create_oval(
            canvas_x - point_radius, canvas_y - point_radius,
            canvas_x + point_radius, canvas_y + point_radius,
            fill="yellow", tags="roi_point"
        )
        if len(self.roi_points) > 1:
            prev_x, prev_y = self.roi_points[-2]
            self.canvas.create_line(
                prev_x, prev_y, canvas_x, canvas_y,
                fill="yellow", width=2, tags="roi_line"
            )

    def update_roi_preview(self, event):
        if not self.drawing_roi or not self.roi_points:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        if self.current_roi_line:
            self.canvas.delete(self.current_roi_line)
        last_x, last_y = self.roi_points[-1]
        self.current_roi_line = self.canvas.create_line(
            last_x, last_y, canvas_x, canvas_y,
            fill="yellow", width=2, dash=(4, 4), tags="roi_preview"
        )

    def complete_roi(self, event):
        if not self.drawing_roi or len(self.roi_points) < 3:
            return
        first_x, first_y = self.roi_points[0]
        last_x, last_y = self.roi_points[-1]
        self.canvas.create_line(
            last_x, last_y, first_x, first_y,
            fill="yellow", width=2, tags="roi_line"
        )
        if self.GUI.current_image:
            image_points = [(x / self.scale, y / self.scale) for x, y in self.roi_points]
            self.roi_polygons[self.GUI.current_image] = image_points
        self.stop_roi_drawing()
        self.draw_roi()
        self.GUI.roi_button.configure(
            text="Edit ROI",
            fg_color=COLORS['secondary']
        )
        self.GUI.update_box_statistics()

    def draw_roi(self):
        self.canvas.delete("roi_line", "roi_point", "roi_preview")
        if not self.GUI.current_image or not self.drawing_roi and \
                self.GUI.current_image not in self.roi_polygons:
            return
        points = self.roi_polygons.get(self.GUI.current_image, [])
        if not points:
            return
        scaled_points = [(x * self.scale, y * self.scale) for x, y in points]
        fill_color = "yellow" if self.hover_roi else ""
        self.canvas.create_polygon(
            *[coord for point in scaled_points for coord in point],
            outline="yellow",
            fill=fill_color,
            stipple="gray50",
            width=2,
            tags="roi_line"
        )
        for x, y in scaled_points:
            self.canvas.create_oval(
                x - 3, y - 3,
                x + 3, y + 3,
                fill="yellow",
                outline="white",
                tags="roi_line"
            )

    def point_in_polygon(self, x, y, polygon):
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_boxes_in_roi(self, threshold=None):
        cls = MODE['cls']
        if not self.GUI.current_image or self.GUI.current_image not in self.roi_polygons:
            if threshold is None:
                return [box for box in self.all_boxes if
                        (box[5] == cls and box[4] >= self.confidence_thresholds[cls]) or
                        (box[5] == 'R' and box[4] >= self.confidence_thresholds['R'])]
            return [box for box in self.all_boxes if box[4] >= threshold]

        roi_points = self.roi_polygons[self.GUI.current_image]
        if not roi_points:
            if threshold is None:
                return [box for box in self.all_boxes if
                        (box[5] == cls and box[4] >= self.confidence_thresholds[cls]) or
                        (box[5] == 'R' and box[4] >= self.confidence_thresholds['R'])]
            return [box for box in self.all_boxes if box[4] >= threshold]

        if threshold is not None:
            threshold_boxes = [box for box in self.all_boxes if box[4] >= threshold]
        else:
            threshold_boxes = [box for box in self.all_boxes if
                               (box[5] == cls and box[4] >= self.confidence_thresholds[cls]) or
                               (box[5] == 'R' and box[4] >= self.confidence_thresholds['R'])]

        roi_boxes = []
        for box in threshold_boxes:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if self.point_in_polygon(center_x, center_y, roi_points):
                roi_boxes.append(box)
        return roi_boxes

    # ── Events ────────────────────────────────────────────────────────────

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<ButtonPress-3>", self.delete_box)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)

    def set_confidence_threshold(self, value, class_name):
        self.confidence_thresholds[class_name] = value
        self.draw_all_boxes()

    def get_visible_boxes(self):
        return [box for box in self.all_boxes
                if box[4] >= self.confidence_thresholds[box[5]]]

    def get_box_at_position(self, x, y):
        for box in self.get_visible_boxes():
            scaled_box = [coord * self.scale for coord in box[:4]]
            corners = [
                (scaled_box[0], scaled_box[1]),
                (scaled_box[2], scaled_box[1]),
                (scaled_box[2], scaled_box[3]),
                (scaled_box[0], scaled_box[3])
            ]
            for corner in corners:
                if abs(x - corner[0]) <= self.corner_radius and abs(y - corner[1]) <= self.corner_radius:
                    return box, 'corner', corners.index(corner)
            if abs(x - scaled_box[0]) <= self.edge_sensitivity and scaled_box[1] <= y <= scaled_box[3]:
                return box, 'edge', 'left'
            if abs(x - scaled_box[2]) <= self.edge_sensitivity and scaled_box[1] <= y <= scaled_box[3]:
                return box, 'edge', 'right'
            if abs(y - scaled_box[1]) <= self.edge_sensitivity and scaled_box[0] <= x <= scaled_box[2]:
                return box, 'edge', 'top'
            if abs(y - scaled_box[3]) <= self.edge_sensitivity and scaled_box[0] <= x <= scaled_box[2]:
                return box, 'edge', 'bottom'
            if (scaled_box[0] <= x <= scaled_box[2] and
                    scaled_box[1] <= y <= scaled_box[3]):
                return box, 'inside', None
        return None, None, None

    def update_cursor(self, hit_area, edge_type=None):
        if hit_area == 'inside':
            self.canvas.configure(cursor="fleur")
        elif hit_area == 'corner':
            self.canvas.configure(cursor="sizing")
        elif hit_area == 'edge':
            if edge_type in ['left', 'right']:
                self.canvas.configure(cursor="sb_h_double_arrow")
            else:
                self.canvas.configure(cursor="sb_v_double_arrow")
        else:
            self.canvas.configure(cursor="arrow")

    def on_mouse_move(self, event):
        if self.is_drawing_new:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        box, hit_area, edge_type = self.get_box_at_position(canvas_x, canvas_y)
        self.update_cursor(hit_area, edge_type)
        if box != self.hover_box:
            self.hover_box = box
            self.draw_all_boxes()

    def on_key_press(self, event):
        cls_key = MODE['cls_key']
        if event.char == 'h' and not self.hide_key_pressed:
            self.hide_key_pressed = True
            self.boxes_hidden = True
            self.draw_all_boxes()
        elif event.char.lower() == cls_key:
            self.selected_class = MODE['cls']
            self.GUI.update_class_selection_status()
        elif event.char == 'r' or event.char == 'R':
            self.selected_class = 'R'
            self.GUI.update_class_selection_status()

    def on_key_release(self, event):
        if event.char == 'h':
            self.hide_key_pressed = False
            self.boxes_hidden = False
            self.draw_all_boxes()

    def on_button_press(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        box, hit_area, edge_type = self.get_box_at_position(canvas_x, canvas_y)
        if box:
            self.active_box = box
            self.box_drag_mode = hit_area
            self.drag_start = (canvas_x, canvas_y)
            if hit_area == 'corner':
                self.resize_handle = edge_type
            elif hit_area == 'edge':
                self.resize_handle = edge_type
        else:
            self.is_drawing_new = True
            self.drag_start = (canvas_x, canvas_y)
            self.current_box = None

    def on_drag(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if self.is_drawing_new:
            if self.current_box:
                self.canvas.delete(self.current_box)
            color = MODE['canvas_color'] if self.selected_class == MODE['cls'] else "red"
            self.current_box = self.canvas.create_rectangle(
                self.drag_start[0], self.drag_start[1],
                canvas_x, canvas_y,
                outline=color, width=2
            )
        elif self.active_box:
            dx = canvas_x - self.drag_start[0]
            dy = canvas_y - self.drag_start[1]
            idx = self.all_boxes.index(self.active_box)
            box = list(self.all_boxes[idx])

            if self.box_drag_mode == 'inside':
                dx_img = dx / self.scale
                dy_img = dy / self.scale
                box[0] += dx_img
                box[1] += dy_img
                box[2] += dx_img
                box[3] += dy_img
            elif self.box_drag_mode in ['edge', 'corner']:
                if self.box_drag_mode == 'corner':
                    corner_idx = self.resize_handle
                    if corner_idx == 0:
                        box[0] += dx / self.scale
                        box[1] += dy / self.scale
                    elif corner_idx == 1:
                        box[2] += dx / self.scale
                        box[1] += dy / self.scale
                    elif corner_idx == 2:
                        box[2] += dx / self.scale
                        box[3] += dy / self.scale
                    elif corner_idx == 3:
                        box[0] += dx / self.scale
                        box[3] += dy / self.scale
                else:
                    edge = self.resize_handle
                    if edge == 'left':
                        box[0] += dx / self.scale
                    elif edge == 'right':
                        box[2] += dx / self.scale
                    elif edge == 'top':
                        box[1] += dy / self.scale
                    elif edge == 'bottom':
                        box[3] += dy / self.scale

            self.all_boxes[idx] = box
            self.active_box = box
            self.drag_start = (canvas_x, canvas_y)
            self.draw_all_boxes()

    def on_button_release(self, event):
        if self.is_drawing_new and self.drag_start:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            min_size = 5
            if (abs(canvas_x - self.drag_start[0]) > min_size and
                    abs(canvas_y - self.drag_start[1]) > min_size):
                x1 = min(self.drag_start[0], canvas_x) / self.scale
                y1 = min(self.drag_start[1], canvas_y) / self.scale
                x2 = max(self.drag_start[0], canvas_x) / self.scale
                y2 = max(self.drag_start[1], canvas_y) / self.scale
                new_box = [x1, y1, x2, y2, 1.0, self.selected_class]
                self.all_boxes.append(new_box)
                self.draw_all_boxes()
                if hasattr(self.GUI, 'update_box_statistics'):
                    self.GUI.update_box_statistics()

        if self.current_box:
            self.canvas.delete(self.current_box)
            self.current_box = None

        if self.active_box and hasattr(self.GUI, 'update_box_statistics'):
            self.GUI.update_box_statistics()

        self.is_drawing_new = False
        self.active_box = None
        self.box_drag_mode = None
        self.resize_handle = None
        self.drag_start = None

    def delete_box(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        box, hit_area, _ = self.get_box_at_position(canvas_x, canvas_y)
        if box:
            self.all_boxes.remove(box)
            self.draw_all_boxes()
            if hasattr(self.GUI, 'update_box_statistics'):
                self.GUI.update_box_statistics()

    def draw_all_boxes(self):
        self.canvas.delete("box")
        if self.boxes_hidden:
            return
        visible_boxes = self.get_visible_boxes()
        cls = MODE['cls']
        for box in visible_boxes:
            scaled_box = [coord * self.scale for coord in box[:4]]
            outline_color = MODE['canvas_color'] if box[5] == cls else "red"
            line_width = 4 if box == self.hover_box else 2
            self.canvas.create_rectangle(
                scaled_box[0], scaled_box[1],
                scaled_box[2], scaled_box[3],
                outline=outline_color, width=line_width, tags="box"
            )
            if box == self.hover_box:
                dot_radius = 3
                corners = [
                    (scaled_box[0], scaled_box[1]),
                    (scaled_box[2], scaled_box[1]),
                    (scaled_box[2], scaled_box[3]),
                    (scaled_box[0], scaled_box[3])
                ]
                for cx, cy in corners:
                    self.canvas.create_oval(
                        cx - dot_radius, cy - dot_radius,
                        cx + dot_radius, cy + dot_radius,
                        fill=outline_color, outline=None, tags="box"
                    )

    # ── Image loading & tiling ────────────────────────────────────────────

    def load_image(self, image_path, boxes=None):
        self.clear_status_message()
        self.loading_label.place(relx=0.5, rely=0.5, anchor='center')
        self.loading_label.lift()
        self.update()

        try:
            self.tile_cache.clear()
            self.canvas.delete("all")
            self.image_path = image_path
            self.original_image = load_image_safely(image_path)
            self.all_boxes = boxes if boxes else []
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            scale_x = canvas_width / self.original_image.width
            scale_y = canvas_height / self.original_image.height
            self.scale = min(scale_x, scale_y)
            self.initial_scale = self.scale
            scaled_width = int(self.original_image.width * self.scale)
            scaled_height = int(self.original_image.height * self.scale)
            self.canvas.configure(scrollregion=(0, 0, scaled_width, scaled_height))
            self.load_visible_tiles()
            self.draw_all_boxes()
        finally:
            self.loading_label.place_forget()

    def get_tile_key(self, row, col, scale):
        return f"{row}_{col}_{scale:.4f}"

    def load_tile(self, row, col):
        if not self.original_image:
            return None
        x1 = col * self.tile_size
        y1 = row * self.tile_size
        x2 = min(x1 + self.tile_size, self.original_image.width)
        y2 = min(y1 + self.tile_size, self.original_image.height)
        tile = self.original_image.crop((x1, y1, x2, y2))
        new_width = max(1, int(tile.width * self.scale))
        new_height = max(1, int(tile.height * self.scale))
        return tile.resize((new_width, new_height), Image.LANCZOS)

    def load_visible_tiles(self):
        if not self.original_image:
            return
        num_rows = math.ceil(self.original_image.height / self.tile_size)
        num_cols = math.ceil(self.original_image.width / self.tile_size)
        for row in range(num_rows):
            for col in range(num_cols):
                tile_key = self.get_tile_key(row, col, self.scale)
                if tile_key not in self.tile_cache:
                    tile = self.load_tile(row, col)
                    if tile:
                        self.tile_cache[tile_key] = ImageTk.PhotoImage(tile)
        self.draw_visible_tiles()

    def draw_visible_tiles(self):
        self.canvas.delete("tile")
        if not self.original_image:
            return
        num_rows = math.ceil(self.original_image.height / self.tile_size)
        num_cols = math.ceil(self.original_image.width / self.tile_size)
        for row in range(num_rows):
            for col in range(num_cols):
                tile_key = self.get_tile_key(row, col, self.scale)
                if tile_key in self.tile_cache:
                    x = col * self.tile_size * self.scale
                    y = row * self.tile_size * self.scale
                    self.canvas.create_image(
                        x, y,
                        image=self.tile_cache[tile_key],
                        anchor="nw",
                        tags="tile"
                    )

    def zoom(self, event):
        if not self.original_image or hasattr(self, '_zooming'):
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        rel_x = canvas_x / (self.original_image.width * self.scale)
        rel_y = canvas_y / (self.original_image.height * self.scale)
        old_scale = self.scale
        target_scale = self.scale * (1.2 if event.delta > 0 else 1 / 1.2)
        min_scale = self.initial_scale * 0.5
        max_scale = self.initial_scale * 7.0
        target_scale = max(min_scale, min(max_scale, target_scale))

        if target_scale != old_scale:
            self._zooming = True

            def prepare_new_view():
                new_cache = {}
                self.scale = target_scale
                scaled_width = int(self.original_image.width * self.scale)
                scaled_height = int(self.original_image.height * self.scale)
                num_rows = math.ceil(self.original_image.height / self.tile_size)
                num_cols = math.ceil(self.original_image.width / self.tile_size)
                for row in range(num_rows):
                    for col in range(num_cols):
                        tile = self.load_tile(row, col)
                        if tile:
                            tile_key = self.get_tile_key(row, col, self.scale)
                            new_cache[tile_key] = ImageTk.PhotoImage(tile)

                def swap_views():
                    self.canvas.configure(scrollregion=(0, 0, scaled_width, scaled_height))
                    self.tile_cache = new_cache
                    view_width = self.canvas.winfo_width()
                    view_height = self.canvas.winfo_height()
                    new_x = rel_x * scaled_width
                    new_y = rel_y * scaled_height
                    frac_x = max(0, min(1, (new_x - view_width / 2) / scaled_width))
                    frac_y = max(0, min(1, (new_y - view_height / 2) / scaled_height))
                    self.canvas.delete("tile")
                    self.draw_visible_tiles()
                    self.draw_all_boxes()
                    self.canvas.xview_moveto(frac_x)
                    self.canvas.yview_moveto(frac_y)
                    del self._zooming

                self.after(1, swap_views)
            self.after(1, prepare_new_view)

        if self.GUI.current_image in self.roi_polygons:
            self.draw_roi()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.draw_visible_tiles()
        self.draw_all_boxes()
        if self.GUI.current_image in self.roi_polygons:
            self.draw_roi()


# ── Main GUI ─────────────────────────────────────────────────────────────────

class ModernDetectionGUI:
    def __init__(self, mode_key):
        self.mode_key = mode_key
        self.root = ctk.CTk()

        cls = MODE['cls']
        cls_label = MODE['cls_label']
        self.root.title("SBeeVia")

        self.root.geometry("1200x800")

        # Initialize models
        self.g_model_path = get_resource_path(os.path.join("model", "weights", MODE['weights']))
        self.r_model_path = get_resource_path(os.path.join("model", "weights", "R.pt"))
        self.g_model = None
        self.r_model = None

        # Initialize variables
        self.current_folder = None
        self.output_path = None
        self.current_image = None
        self.current_boxes = {}
        self.image_confidence_thresholds = {cls: {}, 'R': {}}

        self.image_dimensions = {}

        # Store calculated concentrations
        self.image_concentrations = {}
        self.subfolder_concentrations = {}
        self.total_concentration = 0.0

        # Concentration calculation parameters
        self.chamber_depth = 10
        self.dilution_factor = 1.0
        self.scale = 0.644

        # Set default font
        self.default_font = ("Inter", 13)
        self.header_font = ("Inter", 16, "bold")

        # Setup UI elements
        self.setup_ui()

        # Bind cleanup to window closing
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)

        # Maximize window
        self.root.update()
        self.root.state('zoomed')

    def setup_ui(self):
        cls = MODE['cls']
        cls_label = MODE['cls_label']

        # Create main container
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color=COLORS['surface'],
            corner_radius=15
        )
        self.main_container.pack(fill="both", expand=True, padx=0, pady=0)

        # Create sidebar
        self.sidebar = ctk.CTkFrame(
            self.main_container,
            width=320,
            fg_color=COLORS['surface'],
            corner_radius=10
        )
        self.sidebar.pack(side="left", fill="y", padx=(0, 0), pady=(5, 10))
        self.sidebar.pack_propagate(False)

        # Create right-side container
        self.right_container = ctk.CTkFrame(
            self.main_container,
            fg_color=COLORS['surface'],
            corner_radius=10
        )
        self.right_container.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=0)

        # Header
        self.header_text = ctk.CTkLabel(
            self.sidebar,
            text=f"{cls} and R Viability Analysis",
            font=("Inter", 20, "bold"),
            text_color=COLORS['success']
        )
        self.header_text.pack(pady=(10, 5), padx=20)

        # ── Mode indicator ────────────────────────────────────────────────
        mode_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        mode_frame.pack(fill="x", padx=20, pady=(0, 8))

        self.mode_indicator = ctk.CTkLabel(
            mode_frame,
            text=f"Mode: {cls_label} + Red",
            font=("Inter", 12, "bold"),
            text_color=COLORS['success']
        )
        self.mode_indicator.pack(side="left")

        self.change_mode_btn = ctk.CTkButton(
            mode_frame,
            text="Change mode",
            width=100,
            height=22,
            font=("Inter", 11),
            fg_color=COLORS['secondary'],
            hover_color="#374151",
            corner_radius=6,
            command=self.change_mode
        )
        self.change_mode_btn.pack(side="right")

        # Select folder button
        self.select_button = ctk.CTkButton(
            self.sidebar,
            text="Select Input Folder",
            command=self.run_task_in_thread,
            height=30,
            font=("Inter", 13, "bold"),
            fg_color=COLORS['secondary'],
            hover_color="#374151",
            corner_radius=8
        )
        self.select_button.pack(pady=(0, 5), padx=20, fill="x")

        self.save_results_button = ctk.CTkButton(
            self.sidebar,
            text="Save Results",
            command=self.save_results,
            height=30,
            font=("Inter", 13, "bold"),
            fg_color=COLORS['secondary'],
            hover_color="#374151",
            corner_radius=8,
            state="disabled"
        )
        self.save_results_button.pack(pady=(3, 5), padx=20, fill="x")

        # Class selection frame
        self.class_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.class_frame.pack(fill="x", padx=20, pady=(0, 5))

        self.class_label = ctk.CTkLabel(
            self.class_frame,
            text="Selected Class for Drawing:",
            font=self.default_font,
            text_color=COLORS['text']
        )
        self.class_label.pack(anchor="w", pady=(0, 5))

        self.radio_frame = ctk.CTkFrame(self.class_frame, fg_color="transparent")
        self.radio_frame.pack(fill="x", pady=(0, 0))

        self.class_var = tk.StringVar(value=cls)

        self.class_g_btn = ctk.CTkRadioButton(
            self.radio_frame,
            text=f"Class {cls} ({cls_label})",
            variable=self.class_var,
            value=cls,
            command=self.on_class_selected,
            fg_color=COLORS['success'],
            border_color=COLORS['success'],
            hover_color="#1ba350",
            border_width_checked=8,
        )
        self.class_g_btn.pack(side="left", padx=(0, 10), pady=(0, 0))

        self.class_r_btn = ctk.CTkRadioButton(
            self.radio_frame,
            text="Class R (Red)",
            variable=self.class_var,
            value="R",
            command=self.on_class_selected,
            fg_color=COLORS['error'],
            border_color=COLORS['error'],
            border_width_checked=8,
        )
        self.class_r_btn.pack(side="right", pady=(0, 0))

        # Non-red confidence slider
        self.g_confidence_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.g_confidence_frame.pack(fill="x", padx=20, pady=(0, 5))

        self.g_confidence_label = ctk.CTkLabel(
            self.g_confidence_frame,
            text=f"{cls} Confidence Threshold: 0.10",
            font=self.default_font,
            text_color=COLORS['success']
        )
        self.g_confidence_label.pack(anchor="w", pady=(0, 5))

        self.g_confidence_slider = ctk.CTkSlider(
            self.g_confidence_frame,
            from_=0.01, to=1.0,
            number_of_steps=99,
            command=self.update_g_confidence_threshold,
            height=16,
            button_color=COLORS['success'],
            progress_color=COLORS['success']
        )
        self.g_confidence_slider.set(0.1)
        self.g_confidence_slider.pack(fill="x")

        # R confidence slider
        self.r_confidence_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.r_confidence_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.r_confidence_label = ctk.CTkLabel(
            self.r_confidence_frame,
            text="R Confidence Threshold: 0.10",
            font=self.default_font,
            text_color=COLORS['error']
        )
        self.r_confidence_label.pack(anchor="w", pady=(0, 5))

        self.r_confidence_slider = ctk.CTkSlider(
            self.r_confidence_frame,
            from_=0.01, to=1.0,
            number_of_steps=99,
            command=self.update_r_confidence_threshold,
            height=16,
            button_color=COLORS['error'],
            progress_color=COLORS['error']
        )
        self.r_confidence_slider.set(0.1)
        self.r_confidence_slider.pack(fill="x")

        # Apply to All buttons
        self.apply_g_all_button = ctk.CTkButton(
            self.g_confidence_frame,
            text=f"Apply {cls} threshold to all images",
            command=lambda: self.apply_threshold_to_all(cls),
            height=25,
            font=("Inter", 12, "bold"),
            fg_color=COLORS['success'],
            hover_color=COLORS['primary'],
            corner_radius=8,
            state="disabled"
        )
        self.apply_g_all_button.pack(pady=(10, 0), fill="x")

        self.apply_r_all_button = ctk.CTkButton(
            self.r_confidence_frame,
            text="Apply R threshold to all images",
            command=lambda: self.apply_threshold_to_all('R'),
            height=25,
            font=("Inter", 12, "bold"),
            fg_color=COLORS['error'],
            hover_color=COLORS['primary'],
            corner_radius=8,
            state="disabled"
        )
        self.apply_r_all_button.pack(pady=(10, 0), fill="x")

        # ROI button
        self.roi_button = ctk.CTkButton(
            self.sidebar,
            text="Edit ROI",
            command=self.toggle_roi_mode,
            height=25,
            font=("Inter", 12, "bold"),
            fg_color=COLORS['secondary'],
            hover_color=COLORS['accent'],
            corner_radius=8,
            state="disabled"
        )
        self.roi_button.pack(pady=(0, 10), padx=20, fill="x")

        # Parameters button
        self.params_button = ctk.CTkButton(
            self.sidebar,
            text="Set Concentration Parameters",
            command=self.open_parameters_frame,
            height=25,
            font=("Inter", 12, "bold"),
            fg_color=COLORS['secondary'],
            hover_color=COLORS['accent'],
            corner_radius=8,
            state="disabled"
        )
        self.params_button.pack(pady=(0, 10), padx=20, fill="x")

        # Statistics frame
        self.setup_statistics_frame()

        # Image list
        self.list_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.list_frame.pack(fill="both", expand=True, padx=20)

        self.list_label = ctk.CTkLabel(
            self.list_frame,
            text="Processed Images",
            font=self.header_font,
            text_color=COLORS['text']
        )
        self.list_label.pack(pady=(0, 5), anchor="w")

        self.image_listbox = tk.Listbox(
            self.list_frame,
            bg=COLORS['background'],
            fg=COLORS['text'],
            selectmode="single",
            borderwidth=1,
            highlightthickness=1,
            font=("Inter", 10),
            selectbackground=COLORS['primary'],
            selectforeground=COLORS['background'],
            cursor="hand2",
            selectborderwidth=0,
            activestyle='none',
            height=15
        )
        self.image_listbox.pack(fill="both", expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_select_image)
        self.image_listbox.bind('<Button-3>', self.on_right_click_image)
        self.image_listbox.configure(state="disabled")

        # Image viewer
        self.image_viewer = ModernTiledImageViewer(self.right_container, self)
        self.image_viewer.pack(fill="both", expand=True, padx=0, pady=(0, 5))

        # Status bar
        self.status_bar_frame = ctk.CTkFrame(
            self.right_container,
            fg_color=COLORS['surface'],
            corner_radius=5,
            height=25
        )
        self.status_bar_frame.pack(fill="x", padx=5, pady=(0, 0))
        self.status_bar_frame.pack_propagate(False)

        self.progress_bar = ctk.CTkProgressBar(
            self.status_bar_frame,
            mode="determinate",
            height=5,
            corner_radius=3,
            width=300
        )
        self.progress_bar.place(relx=0.5, rely=0.5, anchor="center")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.status_bar_frame,
            text="Ready",
            font=("Inter", 12),
            text_color=COLORS['secondary'],
            width=150,
            anchor="w"
        )
        progress_bar_half_width = 150
        label_offset = 10
        self.progress_label.place(relx=0.5, rely=0.5, x=progress_bar_half_width + label_offset, anchor="w")

    # ── Change mode ───────────────────────────────────────────────────────

    def change_mode(self):
        """Restart the application in a different mode."""
        has_data = bool(self.current_boxes)
        if has_data:
            ok = messagebox.askyesno(
                "Change mode",
                "Changing mode will discard the current analysis.\n\nDo you want to continue?"
            )
            if not ok:
                return

        # Pick the other mode
        new_key = 'blue' if self.mode_key == 'green' else 'green'
        apply_mode(new_key)

        # Destroy current window and rebuild
        self.root.destroy()
        app = ModernDetectionGUI(new_key)
        app.root.mainloop()

    # ── Parameters dialog ─────────────────────────────────────────────────

    def open_parameters_frame(self):
        params_window = ctk.CTkToplevel(self.root)
        params_window.title("Concentration Parameters")
        params_window.geometry("400x350")
        params_window.resizable(False, False)
        params_window.grab_set()

        params_frame = ctk.CTkFrame(params_window, fg_color=COLORS['surface'])
        params_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = ctk.CTkLabel(
            params_frame,
            text="Concentration Calculation Parameters",
            font=("Inter", 16, "bold"),
            text_color=COLORS['primary']
        )
        title_label.pack(pady=(0, 15))

        # Camera depth
        depth_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        depth_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(depth_frame, text="Camera Depth (microns):",
                      font=self.default_font, text_color=COLORS['text']).pack(side="left", padx=(0, 10))
        depth_entry = ctk.CTkEntry(depth_frame, width=100, placeholder_text="e.g. 0.01")
        depth_entry.insert(0, str(self.chamber_depth))
        depth_entry.pack(side="right")

        # Dilution ratio
        dilution_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        dilution_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(dilution_frame, text="Dilution Ratio (X:Y):",
                      font=self.default_font, text_color=COLORS['text']).pack(side="left", padx=(0, 10))

        ratio_frame = ctk.CTkFrame(dilution_frame, fg_color="transparent")
        ratio_frame.pack(side="right")

        current_x = 1
        current_y = 0
        if hasattr(self, 'dilution_x') and hasattr(self, 'dilution_y'):
            current_x = self.dilution_x
            current_y = self.dilution_y
        elif hasattr(self, 'dilution_factor') and self.dilution_factor > 1:
            current_x = 1
            current_y = self.dilution_factor - 1

        semen_entry = ctk.CTkEntry(ratio_frame, width=45, placeholder_text="X")
        semen_entry.insert(0, str(int(current_x)))
        semen_entry.pack(side="left")

        ctk.CTkLabel(ratio_frame, text=" : ", font=self.default_font,
                      text_color=COLORS['text']).pack(side="left")

        diluent_entry = ctk.CTkEntry(ratio_frame, width=45, placeholder_text="Y")
        diluent_entry.insert(0, str(int(current_y)))
        diluent_entry.pack(side="left")

        # Scale
        scale_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        scale_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(scale_frame, text="Scale (microns/pixel):",
                      font=self.default_font, text_color=COLORS['text']).pack(side="left", padx=(0, 10))
        scale_entry = ctk.CTkEntry(scale_frame, width=100, placeholder_text="e.g. 1.0")
        scale_entry.insert(0, str(self.scale))
        scale_entry.pack(side="right")

        info_text = ("These parameters are used to calculate sperm concentration.\n"
                     "Camera Depth: Optical section depth in microns\n"
                     "Dilution Ratio: X parts semen : Y parts diluent\n"
                     "Scale: Microns per pixel in the image")
        ctk.CTkLabel(params_frame, text=info_text, font=("Inter", 11),
                      text_color=COLORS['secondary'], justify="left").pack(pady=(10, 15))

        calculation_label = ctk.CTkLabel(
            params_frame, text="Dilution Factor: 1.00 ((X+Y)/X)",
            font=("Inter", 11), text_color=COLORS['primary'], justify="center"
        )
        calculation_label.pack(pady=(0, 15))

        def update_calculation(*args):
            try:
                x = float(semen_entry.get() or "1")
                y = float(diluent_entry.get() or "0")
                if x <= 0:
                    calculation_label.configure(text="Error: X must be positive")
                    return
                factor = (x + y) / x
                calculation_label.configure(text=f"Dilution Factor: {factor:.2f} ((X+Y)/X)")
            except ValueError:
                calculation_label.configure(text="Error: Please enter valid numbers")

        semen_entry.bind("<KeyRelease>", update_calculation)
        diluent_entry.bind("<KeyRelease>", update_calculation)
        update_calculation()

        def save_parameters():
            try:
                chamber_depth = float(depth_entry.get())
                if chamber_depth <= 0:
                    messagebox.showerror("Error", "Camera depth must be positive")
                    return
                x = float(semen_entry.get() or "1")
                y = float(diluent_entry.get() or "0")
                if x <= 0:
                    messagebox.showerror("Error", "Semen parts (X) must be positive")
                    return
                dilution_factor = (x + y) / x
                self.dilution_x = x
                self.dilution_y = y
                self.dilution_factor = dilution_factor
                scale_val = float(scale_entry.get())
                if scale_val <= 0:
                    messagebox.showerror("Error", "Scale must be positive")
                    return
                self.chamber_depth = chamber_depth
                self.scale = scale_val
                self.update_box_statistics()
                params_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for all fields")

        ctk.CTkButton(
            params_frame, text="Save Parameters", command=save_parameters,
            height=30, font=("Inter", 12, "bold"),
            fg_color=COLORS['primary'], hover_color=COLORS['accent'], corner_radius=8
        ).pack(pady=(0, 10))

    # ── ROI toggle ────────────────────────────────────────────────────────

    def toggle_roi_mode(self):
        if not hasattr(self.image_viewer, 'drawing_roi') or not self.image_viewer.drawing_roi:
            self.roi_button.configure(
                fg_color=COLORS['primary'],
                text="Editing ROI (click here to exit)"
            )
            self.image_viewer.start_roi_drawing()
        else:
            self.roi_button.configure(
                fg_color=COLORS['secondary'],
                text="Edit ROI"
            )
            self.image_viewer.stop_roi_drawing()

    # ── Statistics ────────────────────────────────────────────────────────

    def setup_statistics_frame(self):
        cls = MODE['cls']

        self.stats_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.stats_frame.pack(fill="x", padx=20, pady=(0, 0))

        self.stats_label = ctk.CTkLabel(
            self.stats_frame, text="Detection Statistics",
            font=self.header_font, text_color=COLORS['text']
        )
        self.stats_label.pack(anchor="w", pady=(0, 0))

        self.g_current_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Current Image ({cls}): 0 objects",
            font=self.default_font, text_color=COLORS['success'], height=20
        )
        self.g_current_boxes_label.pack(anchor="w", pady=(0, 0))

        self.r_current_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text="Current Image (R): 0 objects",
            font=self.default_font, text_color=COLORS['error'], height=20
        )
        self.r_current_boxes_label.pack(anchor="w", pady=(0, 0))

        self.current_concentration_label = ctk.CTkLabel(
            self.stats_frame,
            text="Current Concentration: 0.00 M/ml",
            font=self.default_font, text_color=COLORS['primary'], height=20
        )
        self.current_concentration_label.pack(anchor="w", pady=(0, 0))

        self.g_subfolder_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Subfolder Total ({cls}): 0 objects",
            font=self.default_font, text_color=COLORS['success'], height=20
        )
        self.g_subfolder_boxes_label.pack(anchor="w", pady=(0, 0))

        self.r_subfolder_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text="Subfolder Total (R): 0 objects",
            font=self.default_font, text_color=COLORS['error'], height=20
        )
        self.r_subfolder_boxes_label.pack(anchor="w", pady=(0, 0))

        self.subfolder_concentration_label = ctk.CTkLabel(
            self.stats_frame,
            text="Subfolder Concentration: 0.00 M/ml",
            font=self.default_font, text_color=COLORS['primary'], height=20
        )
        self.subfolder_concentration_label.pack(anchor="w", pady=(0, 0))

        self.g_total_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Total {cls} (all images): 0 objects",
            font=self.default_font, text_color=COLORS['success'], height=20
        )
        self.g_total_boxes_label.pack(anchor="w")

        self.r_total_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text="Total R (all images): 0 objects",
            font=self.default_font, text_color=COLORS['error'], height=20
        )
        self.r_total_boxes_label.pack(anchor="w")

        self.total_concentration_label = ctk.CTkLabel(
            self.stats_frame,
            text="Total Concentration: 0.00 M/ml",
            font=self.default_font, text_color=COLORS['primary'], height=20
        )
        self.total_concentration_label.pack(anchor="w", pady=(0, 0))

    def update_class_selection_status(self):
        self.class_var.set(self.image_viewer.selected_class)

    def on_class_selected(self):
        self.image_viewer.selected_class = self.class_var.get()

    def update_g_confidence_threshold(self, value):
        cls = MODE['cls']
        self.g_confidence_label.configure(text=f"{cls} Confidence Threshold: {value:.2f}")
        if self.current_image:
            self.image_confidence_thresholds[cls][self.current_image] = float(value)
            if hasattr(self, 'image_viewer'):
                self.image_viewer.set_confidence_threshold(value, cls)
            self.update_box_statistics()

    def update_r_confidence_threshold(self, value):
        self.r_confidence_label.configure(text=f"R Confidence Threshold: {value:.2f}")
        if self.current_image:
            self.image_confidence_thresholds['R'][self.current_image] = float(value)
            if hasattr(self, 'image_viewer'):
                self.image_viewer.set_confidence_threshold(value, 'R')
            self.update_box_statistics()

    def apply_threshold_to_all(self, class_name):
        try:
            cls = MODE['cls']
            if class_name == cls:
                current_threshold = self.g_confidence_slider.get()
            else:
                current_threshold = self.r_confidence_slider.get()

            temp_current = self.current_image
            self.update_progress(0, f"Applying {class_name} threshold to all images...")
            total_images = len(self.current_boxes)

            for idx, image_name in enumerate(self.current_boxes.keys()):
                progress = (idx + 1) / total_images
                self.update_progress(progress, f"Updating {class_name} threshold for {image_name}")
                self.image_confidence_thresholds[class_name][image_name] = current_threshold
                if image_name == self.current_image and hasattr(self, 'image_viewer'):
                    self.image_viewer.set_confidence_threshold(current_threshold, class_name)

            self.current_image = temp_current
            self.update_box_statistics()
            self.update_progress(1.0, f"{class_name} threshold applied to all images")
            self.root.after(2000, lambda: self.update_progress(0, "Ready"))

        except Exception as e:
            print(f"Error applying {class_name} threshold to all images: {str(e)}")
            messagebox.showerror("Error", f"Error applying {class_name} threshold: {str(e)}")
            self.update_progress(0, "Ready")

    def update_progress(self, value, text="Processing..."):
        self.progress_bar.set(value)
        self.progress_label.configure(text=text)
        self.root.update_idletasks()

    def get_all_images(self, folder):
        image_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, folder)
                    image_files.append((full_path, rel_path))
        return image_files

    def cleanup(self, exit_program=True):
        try:
            pass
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            if exit_program:
                self.root.destroy()

    def run_task_in_thread(self):
        self.select_button.configure(state="disabled")
        thread = threading.Thread(target=self.select_folder)
        thread.start()

    def select_folder(self):
        cls = MODE['cls']
        try:
            self.image_listbox.configure(state="normal")
            self.image_listbox.delete(0, tk.END)
            self.image_listbox.configure(state="disabled")

            self.current_image = None
            self.current_boxes = {}
            self.image_confidence_thresholds = {cls: {}, 'R': {}}

            self.image_viewer.current_image = None
            self.image_viewer.canvas.delete("all")
            self.image_viewer.original_image = None
            self.image_viewer.all_boxes = []

            self.update_box_statistics()

            if hasattr(self, 'image_viewer'):
                self.image_viewer.canvas.delete("all")

            self.current_folder = filedialog.askdirectory()
            if not self.current_folder:
                self.select_button.configure(state="normal")
                self.image_viewer.show_status_message("Select a folder to begin the analysis")
                return

            self.image_viewer.show_status_message("Analysis in progress…")
            self.process_images()

            self.apply_g_all_button.configure(state="normal")
            self.apply_r_all_button.configure(state="normal")

            if self.current_boxes:
                self.image_viewer.show_status_message("Analysis complete — select an image from the list")
            else:
                self.image_viewer.show_status_message("Select a folder to begin the analysis")

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            messagebox.showerror("Error", f"Error in processing: {str(e)}")
        finally:
            self.select_button.configure(state="normal")

    # ── Statistics calculation ────────────────────────────────────────────

    def update_box_statistics(self):
        cls = MODE['cls']

        g_current_count = 0
        r_current_count = 0
        g_total_count = 0
        r_total_count = 0
        g_subfolder_count = 0
        r_subfolder_count = 0
        current_subfolder = "None"

        if self.current_image:
            current_subfolder = os.path.dirname(self.current_image)
            if current_subfolder == "":
                current_subfolder = "Root"

            g_threshold = self.image_confidence_thresholds[cls].get(self.current_image, 0.1)
            r_threshold = self.image_confidence_thresholds['R'].get(self.current_image, 0.1)

            self.image_viewer.confidence_thresholds[cls] = g_threshold
            self.image_viewer.confidence_thresholds['R'] = r_threshold

            has_roi = self.current_image in self.image_viewer.roi_polygons

            if has_roi:
                visible_boxes = self.image_viewer.get_boxes_in_roi()
            else:
                boxes = self.current_boxes.get(self.current_image, [])
                visible_boxes = [box for box in boxes if
                                 (box[5] == cls and box[4] >= g_threshold) or
                                 (box[5] == 'R' and box[4] >= r_threshold)]

            g_current_count = sum(1 for box in visible_boxes if box[5] == cls)
            r_current_count = sum(1 for box in visible_boxes if box[5] == 'R')

        image_concentrations = {}
        subfolder_images = []

        for image_name, boxes in self.current_boxes.items():
            g_threshold = self.image_confidence_thresholds[cls].get(image_name, 0.1)
            r_threshold = self.image_confidence_thresholds['R'].get(image_name, 0.1)

            has_roi = image_name in self.image_viewer.roi_polygons

            if has_roi:
                temp_current = self.current_image
                temp_boxes = self.image_viewer.all_boxes

                self.current_image = image_name
                self.image_viewer.all_boxes = boxes
                self.image_viewer.confidence_thresholds[cls] = g_threshold
                self.image_viewer.confidence_thresholds['R'] = r_threshold

                filtered_boxes = self.image_viewer.get_boxes_in_roi()

                self.current_image = temp_current
                self.image_viewer.all_boxes = temp_boxes
            else:
                filtered_boxes = [box for box in boxes if
                                  (box[5] == cls and box[4] >= g_threshold) or
                                  (box[5] == 'R' and box[4] >= r_threshold)]

            img_g_count = sum(1 for box in filtered_boxes if box[5] == cls)
            img_r_count = sum(1 for box in filtered_boxes if box[5] == 'R')
            img_total_count = img_g_count + img_r_count

            g_total_count += img_g_count
            r_total_count += img_r_count

            img_subfolder = os.path.dirname(image_name)
            if img_subfolder == current_subfolder or (img_subfolder == "" and current_subfolder == "Root"):
                g_subfolder_count += img_g_count
                r_subfolder_count += img_r_count
                subfolder_images.append(image_name)

            # Concentration calculation
            if has_roi:
                roi_points = self.image_viewer.roi_polygons[image_name]
                if len(roi_points) >= 3:
                    area = 0
                    j = len(roi_points) - 1
                    for i in range(len(roi_points)):
                        area += (roi_points[j][0] + roi_points[i][0]) * (roi_points[j][1] - roi_points[i][1])
                        j = i
                    roi_area_pixels = abs(area) / 2
                    area_sq_microns = roi_area_pixels * (self.scale ** 2)
                else:
                    area_sq_microns = 1000000
            else:
                if image_name in self.image_dimensions:
                    img_width, img_height = self.image_dimensions[image_name]
                    area_pixels = img_width * img_height
                    area_sq_microns = area_pixels * (self.scale ** 2)
                else:
                    area_sq_microns = 1000000

            volume_ml = (area_sq_microns * self.chamber_depth) / 1_000_000_000_000
            if volume_ml > 0 and img_total_count > 0:
                image_concentrations[image_name] = (
                    img_total_count / volume_ml) * self.dilution_factor / 1_000_000
            else:
                image_concentrations[image_name] = 0.0

        # Percentages
        current_total = g_current_count + r_current_count
        g_current_percentage = 0 if current_total == 0 else (g_current_count / current_total) * 100

        subfolder_total = g_subfolder_count + r_subfolder_count
        g_subfolder_percentage = 0 if subfolder_total == 0 else (g_subfolder_count / subfolder_total) * 100

        grand_total = g_total_count + r_total_count
        g_total_percentage = 0 if grand_total == 0 else (g_total_count / grand_total) * 100

        roi_text = " (in ROI)" if self.current_image and self.current_image in self.image_viewer.roi_polygons else ""

        current_conc = image_concentrations.get(self.current_image, 0.0) if self.current_image else 0.0

        subfolder_conc = 0.0
        if subfolder_images:
            subfolder_concentrations = [image_concentrations.get(img, 0.0) for img in subfolder_images]
            subfolder_conc = sum(subfolder_concentrations) / len(subfolder_concentrations)

        total_conc = 0.0
        if image_concentrations:
            total_conc = sum(image_concentrations.values()) / len(image_concentrations)

        # Update labels
        self.g_current_boxes_label.configure(
            text=f"Current Image{roi_text} ({cls}): {g_current_count} objects ({g_current_percentage:.1f}%)")
        self.r_current_boxes_label.configure(
            text=f"Current Image{roi_text} (R): {r_current_count} objects ({100 - g_current_percentage:.1f}%)")

        self.g_subfolder_boxes_label.configure(
            text=f"Subfolder Total ({cls}): {g_subfolder_count} objects ({g_subfolder_percentage:.1f}%)")
        self.r_subfolder_boxes_label.configure(
            text=f"Subfolder Total (R): {r_subfolder_count} objects ({100 - g_subfolder_percentage:.1f}%)")

        self.g_total_boxes_label.configure(
            text=f"Total {cls} (all images): {g_total_count} objects ({g_total_percentage:.1f}%)")
        self.r_total_boxes_label.configure(
            text=f"Total R (all images): {r_total_count} objects ({100 - g_total_percentage:.1f}%)")

        self.current_concentration_label.configure(
            text=f"Current Concentration{roi_text}: {current_conc:,.2f} M/ml")
        self.subfolder_concentration_label.configure(
            text=f"Subfolder Concentration: {subfolder_conc:,.2f} M/ml")
        self.total_concentration_label.configure(
            text=f"Total Concentration: {total_conc:,.2f} M/ml")

        self.image_viewer.draw_all_boxes()
        if self.current_image in self.image_viewer.roi_polygons:
            self.image_viewer.draw_roi()

        self.image_concentrations[self.current_image] = current_conc if self.current_image else 0.0

        if current_subfolder not in self.subfolder_concentrations:
            self.subfolder_concentrations = {}
        self.subfolder_concentrations[current_subfolder] = subfolder_conc
        self.total_concentration = total_conc

    # ── Save results ──────────────────────────────────────────────────────

    def save_results(self):
        cls = MODE['cls']
        if not self.current_folder or not self.current_boxes:
            messagebox.showwarning("Warning", "No processed images to save")
            return

        try:
            self.update_progress(0, "Preparing to save results...")
            self.update_progress(0.1, "Calculating concentrations for all images...")

            temp_current_image = self.current_image
            temp_all_boxes = self.image_viewer.all_boxes if hasattr(self.image_viewer, 'all_boxes') else []

            self.image_concentrations = {}

            for image_name in self.current_boxes.keys():
                if image_name not in self.image_concentrations:
                    self.current_image = image_name
                    self.image_viewer.all_boxes = self.current_boxes[image_name]

                    g_threshold = self.image_confidence_thresholds[cls].get(image_name, 0.1)
                    r_threshold = self.image_confidence_thresholds['R'].get(image_name, 0.1)
                    self.image_viewer.confidence_thresholds[cls] = g_threshold
                    self.image_viewer.confidence_thresholds['R'] = r_threshold

                    self.update_box_statistics()

            self.current_image = temp_current_image
            self.image_viewer.all_boxes = temp_all_boxes
            self.update_box_statistics()

            results_dir = os.path.join(self.current_folder, "Results")
            images_dir = os.path.join(results_dir, "images")
            labels_dir = os.path.join(results_dir, "labels")

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            stats_data = []
            stats_header = ["filename", f"threshold_{cls}", "threshold_R",
                            f"{cls}_count", "R_count", "total_count",
                            f"{cls}_percentage", "has_roi", "concentration_M_ml"]

            subfolder_data = {}

            # First pass: count visible boxes
            for image_name, boxes in self.current_boxes.items():
                subfolder_name = os.path.dirname(image_name)
                if subfolder_name == '':
                    subfolder_name = '.'

                if subfolder_name not in subfolder_data:
                    subfolder_data[subfolder_name] = {
                        'images': [],
                        f'{cls}_count': 0,
                        'R_count': 0,
                        'ROI_images': 0,
                        'concentrations': []
                    }

                subfolder_data[subfolder_name]['images'].append(os.path.basename(image_name))

                if image_name in self.image_concentrations:
                    subfolder_data[subfolder_name]['concentrations'].append(
                        self.image_concentrations[image_name])

                g_threshold = self.image_confidence_thresholds[cls].get(image_name, 0.1)
                r_threshold = self.image_confidence_thresholds['R'].get(image_name, 0.1)

                has_roi = image_name in self.image_viewer.roi_polygons

                if has_roi:
                    temp_current = self.current_image
                    temp_boxes = self.image_viewer.all_boxes

                    self.current_image = image_name
                    self.image_viewer.all_boxes = boxes
                    self.image_viewer.confidence_thresholds[cls] = g_threshold
                    self.image_viewer.confidence_thresholds['R'] = r_threshold

                    visible_boxes = self.image_viewer.get_boxes_in_roi()

                    self.current_image = temp_current
                    self.image_viewer.all_boxes = temp_boxes
                else:
                    visible_boxes = [box for box in boxes if
                                     (box[5] == cls and box[4] >= g_threshold) or
                                     (box[5] == 'R' and box[4] >= r_threshold)]

                g_count = sum(1 for box in visible_boxes if box[5] == cls)
                r_count = sum(1 for box in visible_boxes if box[5] == 'R')

                subfolder_name = os.path.dirname(image_name)
                if subfolder_name == '':
                    subfolder_name = '.'

                if subfolder_name not in subfolder_data:
                    subfolder_data[subfolder_name] = {
                        'images': [],
                        f'{cls}_count': 0,
                        'R_count': 0,
                        'ROI_images': 0
                    }

                subfolder_data[subfolder_name]['images'].append(os.path.basename(image_name))
                subfolder_data[subfolder_name][f'{cls}_count'] += g_count
                subfolder_data[subfolder_name]['R_count'] += r_count
                if has_roi:
                    subfolder_data[subfolder_name]['ROI_images'] += 1

            # Process each image
            total_files = len(self.current_boxes)
            for idx, (image_name, boxes) in enumerate(self.current_boxes.items()):
                progress = (idx + 1) / (total_files * 2)
                self.update_progress(progress, f"Processing {idx + 1}/{total_files}: {image_name}")

                original_image_path = os.path.join(self.current_folder, image_name)

                g_threshold = self.image_confidence_thresholds[cls].get(image_name, 0.1)
                r_threshold = self.image_confidence_thresholds['R'].get(image_name, 0.1)

                has_roi = image_name in self.image_viewer.roi_polygons

                if has_roi:
                    temp_current = self.current_image
                    temp_boxes = self.image_viewer.all_boxes

                    self.current_image = image_name
                    self.image_viewer.all_boxes = boxes
                    self.image_viewer.confidence_thresholds[cls] = g_threshold
                    self.image_viewer.confidence_thresholds['R'] = r_threshold

                    visible_boxes = self.image_viewer.get_boxes_in_roi()

                    self.current_image = temp_current
                    self.image_viewer.all_boxes = temp_boxes
                else:
                    visible_boxes = [box for box in boxes if
                                     (box[5] == cls and box[4] >= g_threshold) or
                                     (box[5] == 'R' and box[4] >= r_threshold)]

                g_count = sum(1 for box in visible_boxes if box[5] == cls)
                r_count = sum(1 for box in visible_boxes if box[5] == 'R')
                total_count = g_count + r_count

                g_percentage = 0.0 if total_count == 0 else (g_count / total_count * 100)

                concentration = self.image_concentrations.get(image_name, 0.0)

                stats_data.append([
                    image_name,
                    g_threshold,
                    r_threshold,
                    g_count,
                    r_count,
                    total_count,
                    round(g_percentage, 2),
                    "Yes" if has_roi else "No",
                    round(concentration, 2)
                ])

                self.save_image_with_boxes(original_image_path, visible_boxes, images_dir)
                self.save_yolo_labels(original_image_path, visible_boxes, labels_dir)

            # Save statistics CSV
            progress = 0.5
            self.update_progress(progress, "Saving statistics...")

            csv_path = os.path.join(results_dir, "statistics.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(stats_header)
                writer.writerows(stats_data)

            # Save subfolder statistics
            progress = 0.75
            self.update_progress(progress, "Saving subfolder statistics...")

            subfolder_stats_data = []
            subfolder_stats_header = ["subfolder", "image_count", "image_list",
                                      f"{cls}_count", "R_count",
                                      "total_detections", f"{cls}_percentage",
                                      "images_with_roi", "concentration_M_ml"]

            for subfolder_name, data in subfolder_data.items():
                g_count = data[f'{cls}_count']
                r_count = data['R_count']
                total_detections = g_count + r_count
                g_percentage = 0.0 if total_detections == 0 else (g_count / total_detections * 100)

                subfolder_concentration = 0.0
                if 'concentrations' in data and data['concentrations']:
                    subfolder_concentration = sum(data['concentrations']) / len(data['concentrations'])

                self.subfolder_concentrations[subfolder_name] = subfolder_concentration

                image_list = ", ".join(data['images'])
                subfolder_name_safe = subfolder_name.replace("-", "_")

                subfolder_stats_data.append([
                    subfolder_name_safe,
                    len(data['images']),
                    image_list,
                    g_count,
                    r_count,
                    total_detections,
                    round(g_percentage, 2),
                    data['ROI_images'],
                    round(subfolder_concentration, 2)
                ])

            subfolder_csv_path = os.path.join(results_dir, "statistics_subfolders.csv")
            with open(subfolder_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(subfolder_stats_header)
                writer.writerows(subfolder_stats_data)

            self.update_progress(1.0, "Results saved successfully")
            messagebox.showinfo("Success", f"Results saved to {results_dir}")
            self.root.after(2000, lambda: self.update_progress(0, "Ready"))

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            messagebox.showerror("Error", f"Error saving results: {str(e)}")
            self.update_progress(0, "Ready")

    def save_image_with_boxes(self, image_path, boxes, output_dir):
        cls = MODE['cls']
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return

            for box in boxes:
                x1, y1, x2, y2, conf, class_name = box
                color = MODE['bgr_color'] if class_name == cls else (0, 0, 255)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            g_count = sum(1 for box in boxes if box[5] == cls)
            r_count = sum(1 for box in boxes if box[5] == 'R')

            img_rel_path = os.path.relpath(image_path, self.current_folder)

            roi_text = ""
            if img_rel_path in self.image_viewer.roi_polygons:
                roi_text = " (in ROI)"
                points = self.image_viewer.roi_polygons[img_rel_path]
                points = [(int(x), int(y)) for x, y in points]
                overlay = img.copy()
                cv2.fillPoly(overlay, [np.array(points)], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
                cv2.polylines(img, [np.array(points)], True, (0, 255, 255), 2)

            info_text = f"{cls}: {g_count} | R: {r_count}{roi_text}"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            padding = 10
            box_width = text_size[0] + padding * 2
            box_height = text_size[1] + padding * 2

            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.putText(img, info_text, (10 + padding, 10 + box_height - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            rel_path = os.path.normpath(os.path.dirname(os.path.relpath(image_path, self.current_folder)))
            if rel_path != '.':
                img_output_dir = os.path.join(output_dir, rel_path)
                os.makedirs(img_output_dir, exist_ok=True)
            else:
                img_output_dir = output_dir

            img_filename = os.path.basename(image_path)
            output_path = os.path.join(img_output_dir, img_filename)
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"Error saving image with boxes: {str(e)}")

    def get_visible_boxes_for_image(self, boxes, g_threshold, r_threshold):
        cls = MODE['cls']
        return [
            box for box in boxes if
            (box[5] == cls and box[4] >= g_threshold) or
            (box[5] == 'R' and box[4] >= r_threshold)
        ]

    def save_yolo_labels(self, image_path, boxes, output_dir):
        cls = MODE['cls']
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return

            img_h, img_w = img.shape[:2]

            rel_path = os.path.normpath(os.path.dirname(os.path.relpath(image_path, self.current_folder)))
            if rel_path != '.':
                label_output_dir = os.path.join(output_dir, rel_path)
                os.makedirs(label_output_dir, exist_ok=True)
            else:
                label_output_dir = output_dir

            img_filename = os.path.basename(image_path)
            base_name = os.path.splitext(img_filename)[0]
            label_path = os.path.join(label_output_dir, f"{base_name}.txt")

            img_rel_path = os.path.relpath(image_path, self.current_folder)

            if img_rel_path in self.image_viewer.roi_polygons:
                temp_current = self.current_image
                temp_boxes = self.image_viewer.all_boxes

                self.current_image = img_rel_path
                self.image_viewer.all_boxes = boxes

                g_threshold = self.image_confidence_thresholds[cls].get(img_rel_path, 0.1)
                r_threshold = self.image_confidence_thresholds['R'].get(img_rel_path, 0.1)
                self.image_viewer.confidence_thresholds[cls] = g_threshold
                self.image_viewer.confidence_thresholds['R'] = r_threshold

                roi_boxes = self.image_viewer.get_boxes_in_roi()

                self.current_image = temp_current
                self.image_viewer.all_boxes = temp_boxes

                boxes = roi_boxes

            with open(label_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2, _, class_name = box
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    class_idx = 0 if class_name == cls else 1 if class_name == 'R' else 2
                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")

        except Exception as e:
            print(f"Error saving YOLO labels: {str(e)}")

    # ── Detection ─────────────────────────────────────────────────────────

    def process_images(self):
        cls = MODE['cls']
        file_images = self.get_all_images(self.current_folder)

        if not file_images:
            messagebox.showwarning("Warning", "No JPG images found in selected folder or subfolders")
            return

        if self.g_model is None:
            try:
                self.update_progress(0, f"Loading {cls} model...")
                self.g_model = YOLO(self.g_model_path)
                self.update_progress(0.25, f"{cls} model loaded successfully")
            except Exception as e:
                print(f"Error loading {cls} model: {str(e)}")
                messagebox.showerror("Error", f"Error loading {cls} model: {str(e)}")
                return

        if self.r_model is None:
            try:
                self.update_progress(0.5, "Loading R model...")
                self.r_model = YOLO(self.r_model_path)
                self.update_progress(0.75, "R model loaded successfully")
            except Exception as e:
                print(f"Error loading R model: {str(e)}")
                messagebox.showerror("Error", f"Error loading R model: {str(e)}")
                return

        self.update_progress(1.0, "Models loaded, running detection...")
        self.run_detection(file_images)

        self.save_results_button.configure(state="normal")
        self.params_button.configure(state="normal")

    def run_detection(self, file_images):
        cls = MODE['cls']
        total_files = len(file_images)

        if total_files == 0:
            self.update_progress(0, "No images to process")
            return

        for idx, (img_path, rel_path) in enumerate(file_images, 1):
            try:
                progress = idx / total_files
                self.update_progress(progress, f"Analyzing image {idx}/{total_files}: {rel_path}")

                g_results = self.g_model(img_path, conf=0.01, iou=0.4, verbose=False)
                r_results = self.r_model(img_path, conf=0.01, iou=0.4, verbose=False)

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                img_h, img_w = img.shape[:2]
                self.image_dimensions[rel_path] = (img_w, img_h)

                boxes = []

                for result in g_results:
                    boxes_data = result.boxes
                    for i in range(len(boxes_data)):
                        confidence = float(boxes_data.conf[i])
                        x, y, w, h = boxes_data.xywhn[i].tolist()
                        x1 = int((x - w / 2) * img_w)
                        y1 = int((y - h / 2) * img_h)
                        x2 = int((x + w / 2) * img_w)
                        y2 = int((y + h / 2) * img_h)
                        boxes.append([x1, y1, x2, y2, confidence, cls])

                for result in r_results:
                    boxes_data = result.boxes
                    for i in range(len(boxes_data)):
                        confidence = float(boxes_data.conf[i])
                        x, y, w, h = boxes_data.xywhn[i].tolist()
                        x1 = int((x - w / 2) * img_w)
                        y1 = int((y - h / 2) * img_h)
                        x2 = int((x + w / 2) * img_w)
                        y2 = int((y + h / 2) * img_h)
                        boxes.append([x1, y1, x2, y2, confidence, 'R'])

                self.current_boxes[rel_path] = boxes

                self.image_confidence_thresholds[cls][rel_path] = 0.1
                self.image_confidence_thresholds['R'][rel_path] = 0.1

                g_count = sum(1 for box in boxes if box[5] == cls)
                r_count = sum(1 for box in boxes if box[5] == 'R')
                print(f"Detected in {rel_path}: {g_count} {cls} objects, {r_count} R objects")

            except Exception as e:
                print(f"Error analyzing image {rel_path}: {str(e)}")

        self.image_listbox.configure(state="normal")
        self.update_image_list()
        self.update_box_statistics()

        self.update_progress(1.0, "Detection complete")
        self.root.after(2000, lambda: self.update_progress(0, "Ready"))

    # ── Image list & selection ────────────────────────────────────────────

    def update_image_list(self):
        self.image_listbox.delete(0, tk.END)
        if self.current_boxes:
            max_width = 50
            self.image_listbox.fullnames = {}
            for image_path in sorted(self.current_boxes.keys()):
                if len(image_path) > max_width:
                    display_name = "..." + image_path[-(max_width - 3):]
                else:
                    display_name = image_path
                padded_file = f"  {display_name}  "
                self.image_listbox.insert(tk.END, padded_file)
                self.image_listbox.fullnames[display_name] = image_path
            self.image_listbox.configure(state="normal")

    def highlight_same_folder_images(self):
        if not self.current_image:
            return
        current_folder = os.path.dirname(self.current_image)
        for i in range(self.image_listbox.size()):
            self.image_listbox.itemconfig(i, {'bg': COLORS['background']})
        for i in range(self.image_listbox.size()):
            item = self.image_listbox.get(i).strip()
            if item:
                full_name = self.image_listbox.fullnames.get(item.strip(), item.strip())
                if os.path.dirname(full_name) == current_folder:
                    self.image_listbox.itemconfig(i, {'bg': '#E3F2FD'})

    def on_select_image(self, event):
        cls = MODE['cls']
        if not self.image_listbox.curselection():
            return

        selected_display = self.image_listbox.get(self.image_listbox.curselection()).strip()
        if not selected_display:
            return

        selected = self.image_listbox.fullnames.get(selected_display.strip(), selected_display.strip())
        if not selected:
            return

        try:
            self.current_image = selected
            self.highlight_same_folder_images()

            image_path = os.path.join(self.current_folder, selected)
            if not os.path.exists(image_path):
                messagebox.showerror("Error", f"Image file not found: {image_path}")
                return

            boxes = self.current_boxes.get(selected, [])
            self.image_viewer.load_image(image_path, boxes)

            g_threshold = self.image_confidence_thresholds[cls].get(selected, 0.1)
            r_threshold = self.image_confidence_thresholds['R'].get(selected, 0.1)

            self.g_confidence_slider.set(g_threshold)
            self.r_confidence_slider.set(r_threshold)

            self.g_confidence_label.configure(text=f"{cls} Confidence Threshold: {g_threshold:.2f}")
            self.r_confidence_label.configure(text=f"R Confidence Threshold: {r_threshold:.2f}")

            self.image_viewer.set_confidence_threshold(g_threshold, cls)
            self.image_viewer.set_confidence_threshold(r_threshold, 'R')

            self.image_viewer.selected_class = self.class_var.get()

            if self.current_image in self.image_viewer.roi_polygons:
                self.image_viewer.draw_roi()

            self.roi_button.configure(state="normal")

            if hasattr(self.image_viewer, 'drawing_roi') and self.image_viewer.drawing_roi:
                self.image_viewer.stop_roi_drawing()
                self.roi_button.configure(
                    fg_color=COLORS['secondary'],
                    text="Edit ROI"
                )

            self.update_box_statistics()

        except Exception as e:
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def on_right_click_image(self, event):
        cls = MODE['cls']
        try:
            clicked_index = self.image_listbox.nearest(event.y)
            if clicked_index < 0 or clicked_index >= self.image_listbox.size():
                return

            selected_display = self.image_listbox.get(clicked_index).strip()
            if not selected_display:
                return

            image_to_delete = self.image_listbox.fullnames.get(selected_display)
            if not image_to_delete:
                return

            confirmation = messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to remove the image '{image_to_delete}' from the analysis?\n\nThis action cannot be undone."
            )

            if confirmation:
                if image_to_delete in self.current_boxes:
                    del self.current_boxes[image_to_delete]
                if image_to_delete in self.image_confidence_thresholds[cls]:
                    del self.image_confidence_thresholds[cls][image_to_delete]
                if image_to_delete in self.image_confidence_thresholds['R']:
                    del self.image_confidence_thresholds['R'][image_to_delete]
                if image_to_delete in self.image_dimensions:
                    del self.image_dimensions[image_to_delete]
                if image_to_delete in self.image_viewer.roi_polygons:
                    del self.image_viewer.roi_polygons[image_to_delete]
                if image_to_delete in self.image_concentrations:
                    del self.image_concentrations[image_to_delete]

                if self.current_image == image_to_delete:
                    self.current_image = None
                    self.image_viewer.canvas.delete("all")
                    self.image_viewer.original_image = None
                    self.image_viewer.all_boxes = []
                    self.roi_button.configure(state="disabled")
                    self.image_viewer.show_status_message("Analysis complete — select an image from the list")

                self.update_image_list()
                self.update_box_statistics()

                if not self.current_boxes:
                    self.save_results_button.configure(state="disabled")
                    self.params_button.configure(state="disabled")
                    self.apply_g_all_button.configure(state="disabled")
                    self.apply_r_all_button.configure(state="disabled")
                    self.image_listbox.configure(state="disabled")
                    self.image_viewer.show_status_message("Select a folder to begin the analysis")

        except Exception as e:
            print(f"Error during image deletion: {e}")
            messagebox.showerror("Error", f"An error occurred while trying to delete the image: {e}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    mode_key = choose_mode()
    if mode_key is None:
        return  # User closed the dialog

    apply_mode(mode_key)

    app = ModernDetectionGUI(mode_key)
    app.root.mainloop()


if __name__ == "__main__":
    main()

# pyinstaller --onefile --noconsole --add-data "model/weights/G.pt;model/weights" --add-data "model/weights/B.pt;model/weights" --add-data "model/weights/R.pt;model/weights" SBeeVia.py --name=SBeeVia