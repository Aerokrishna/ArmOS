#!/usr/bin/env python3
"""Native desktop GUI for commanding one Armos arm over ROS 2."""

import json
import csv
import os
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
from pathlib import Path

import rclpy
from armos_interfaces.msg import ArmosCartControl, ArmosJointControl, ArmosTelemetry
from rclpy.node import Node


class ArmosControlGui(Node):
    """ROS node and Tk interface for joint and Cartesian jogging."""

    def __init__(self, root: tk.Tk):
        super().__init__("armos_control_gui")
        self.root = root
        self.root.title("Armos Control Desk")
        self.root.geometry("980x900")
        self.root.minsize(820, 760)
        self.root.configure(bg="#101820")

        self._joint_pub = None
        self._cart_pub = None
        self._telemetry_sub = None
        self._enabled = tk.BooleanVar(value=False)
        self._logging = tk.BooleanVar(value=False)
        self._log_button_text = tk.StringVar(value="LOG OFF")
        self._ui_active_text = tk.StringVar(value="UI INACTIVE")
        self._robot_id = tk.StringVar(value="left")
        self._status = tk.StringVar(value="Output disabled")
        self._pose_name = tk.StringVar()
        self._selected_pose = tk.StringVar()
        self._step_size = tk.DoubleVar(value=0.02)
        self._angular_step_size = tk.DoubleVar(value=0.10)
        self._jogging = {}
        self._joint_values = [tk.DoubleVar(value=0.0) for _ in range(7)]
        self._telemetry_lock = threading.Lock()
        self._latest_telemetry = None
        self._telemetry_rows = []
        self._logging_active = False
        self._pose_file = Path.home() / ".config" / "armos" / "joint_configurations.json"
        self._poses = self._load_poses()

        self._configure_style()
        self._build_ui()
        self._set_robot_id()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#101820")
        style.configure("Panel.TFrame", background="#17232d")
        style.configure("Title.TLabel", background="#101820", foreground="#f1f5f2",
                        font=("DejaVu Sans", 22, "bold"))
        style.configure("Muted.TLabel", background="#101820", foreground="#8fa3ad",
                        font=("DejaVu Sans", 10))
        style.configure("PanelTitle.TLabel", background="#17232d", foreground="#f1f5f2",
                        font=("DejaVu Sans", 13, "bold"))
        style.configure("PanelText.TLabel", background="#17232d", foreground="#a8bbc3",
                        font=("DejaVu Sans", 10))
        style.configure("Value.TLabel", background="#17232d", foreground="#f2c14e",
                        font=("DejaVu Sans Mono", 10, "bold"))
        style.configure("Action.TButton", background="#243844", foreground="#f1f5f2",
                        borderwidth=0, padding=(12, 8), font=("DejaVu Sans", 10, "bold"))
        style.map("Action.TButton", background=[("active", "#315468")])
        style.configure("CartAction.TButton", background="#243844", foreground="#f1f5f2",
                borderwidth=0, padding=(8, 4), font=("DejaVu Sans", 9, "bold"))
        style.map("CartAction.TButton", background=[("active", "#315468")])
        style.configure("UiInactive.TButton", background="#c98b2e", foreground="#101820",
                borderwidth=0, padding=(24, 15), font=("DejaVu Sans", 13, "bold"))
        style.map("UiInactive.TButton", background=[("active", "#e0a743")])
        style.configure("UiActive.TButton", background="#42b883", foreground="#101820",
                borderwidth=0, padding=(24, 15), font=("DejaVu Sans", 13, "bold"))
        style.map("UiActive.TButton", background=[("active", "#65d29e")])
        style.configure("Stop.TButton", background="#b84444", foreground="#ffffff",
                borderwidth=0, padding=(24, 15), font=("DejaVu Sans", 13, "bold"))
        style.map("Stop.TButton", background=[("active", "#d45d5d")])
        style.configure("LogOff.TButton", background="#536875", foreground="#ffffff",
                borderwidth=0, padding=(12, 8), font=("DejaVu Sans", 10, "bold"))
        style.map("LogOff.TButton", background=[("active", "#6b8594")])
        style.configure("LogOn.TButton", background="#e09b3d", foreground="#101820",
                borderwidth=0, padding=(12, 8), font=("DejaVu Sans", 10, "bold"))
        style.map("LogOn.TButton", background=[("active", "#f0b15c")])
        style.configure("Horizontal.TScale", background="#17232d", troughcolor="#293b46",
                        sliderthickness=18)

    def _build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=28)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 22))
        ttk.Label(header, text="ARMOS CONTROL DESK", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Native ROS 2 command station", style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

        connection = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        connection.pack(fill="x", pady=(0, 16))
        ttk.Label(connection, text="ROBOT LINK", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(connection, text="Robot ID", style="PanelText.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 0))
        robot_entry = ttk.Entry(connection, textvariable=self._robot_id, width=22, font=("DejaVu Sans", 11))
        robot_entry.grid(row=1, column=1, sticky="w", padx=(12, 10), pady=(14, 0))
        robot_entry.bind("<Return>", lambda _event: self._set_robot_id())
        ttk.Button(connection, text="Connect", style="Action.TButton",
                   command=self._set_robot_id).grid(row=1, column=2, sticky="ew",
                                                    padx=(8, 8), pady=(14, 0))
        self._active_button = ttk.Button(connection, textvariable=self._ui_active_text,
                                         style="UiInactive.TButton", command=self._toggle_enabled)
        self._active_button.grid(row=1, column=3, sticky="ew", padx=(8, 8), pady=(14, 0))
        ttk.Button(connection, text="STOP ROBOT", style="Stop.TButton", command=self._stop_robot).grid(
            row=1, column=4, sticky="ew", padx=(8, 0), pady=(14, 0))
        connection.columnconfigure(2, weight=1)
        connection.columnconfigure(3, weight=1)
        connection.columnconfigure(4, weight=1)
        ttk.Label(connection, textvariable=self._status, style="PanelText.TLabel").grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(12, 0))
        telemetry_actions = ttk.Frame(connection, style="Panel.TFrame")
        telemetry_actions.grid(row=3, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        ttk.Button(telemetry_actions, text="REFRESH JOINTS", style="Action.TButton",
                   command=self._refresh_joint_positions).pack(side="left")
        self._log_button = ttk.Button(telemetry_actions, textvariable=self._log_button_text,
                                      style="LogOff.TButton", command=self._toggle_logging)
        self._log_button.pack(side="left", padx=(8, 0))

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True)
        joints = ttk.Frame(body, style="Panel.TFrame", padding=18)
        joints.pack(side="left", fill="both", expand=True, padx=(0, 8))
        cart = ttk.Frame(body, style="Panel.TFrame", padding=18)
        cart.pack(side="right", fill="both", expand=True, padx=(8, 0))
        self._build_joint_panel(joints)
        self._build_cart_panel(cart)

    def _build_joint_panel(self, panel):
        ttk.Label(panel, text="JOINT TARGETS", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Position mode | radians", style="PanelText.TLabel").pack(anchor="w", pady=(3, 18))
        pose_row = ttk.Frame(panel, style="Panel.TFrame")
        pose_row.pack(fill="x", pady=(0, 14))
        ttk.Entry(pose_row, textvariable=self._pose_name, width=14).pack(side="left")
        ttk.Button(pose_row, text="Save", style="Action.TButton",
                   command=self._save_pose).pack(side="left", padx=(6, 0))
        ttk.Label(pose_row, text="Saved", style="PanelText.TLabel").pack(side="left", padx=(12, 5))
        self._pose_dropdown = ttk.Combobox(pose_row, textvariable=self._selected_pose,
                                           state="readonly", width=16)
        self._pose_dropdown.pack(side="left", fill="x", expand=True)
        ttk.Button(pose_row, text="Move", style="Action.TButton",
                   command=self._move_to_pose).pack(side="left", padx=(6, 0))
        for index, value in enumerate(self._joint_values):
            row = ttk.Frame(panel, style="Panel.TFrame")
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=f"J{index + 1}", width=4, style="PanelText.TLabel").pack(side="left")
            ttk.Scale(row, from_=-3.14, to=3.14, variable=value, orient="horizontal",
                      command=lambda _value: self._publish_joint()).pack(side="left", fill="x", expand=True, padx=8)
            ttk.Label(row, textvariable=value, width=7, style="Value.TLabel", anchor="e").pack(side="right")

    def _build_cart_panel(self, panel):
        ttk.Label(panel, text="CARTESIAN JOG", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Hold a button to jog the TCP", style="PanelText.TLabel").pack(anchor="w", pady=(3, 12))
        step_row = ttk.Frame(panel, style="Panel.TFrame")
        step_row.pack(fill="x", pady=(0, 16))
        ttk.Label(step_row, text="Linear speed", style="PanelText.TLabel").pack(side="left")
        ttk.Scale(step_row, from_=0.005, to=0.10, variable=self._step_size,
                  orient="horizontal").pack(side="left", fill="x", expand=True, padx=12)
        ttk.Label(step_row, textvariable=self._step_size, width=6, style="Value.TLabel").pack(side="right")

        self._build_jog_column(panel, "LINEAR", (
            ("+X", "x+"), ("-X", "x-"), ("+Y", "y+"),
            ("-Y", "y-"), ("+Z", "z+"), ("-Z", "z-")), self._step_size)

        angular_row = ttk.Frame(panel, style="Panel.TFrame")
        angular_row.pack(fill="x", pady=(18, 0))
        ttk.Label(angular_row, text="Angular speed", style="PanelText.TLabel").pack(side="left")
        ttk.Scale(angular_row, from_=0.02, to=0.50, variable=self._angular_step_size,
                  orient="horizontal").pack(side="left", fill="x", expand=True, padx=12)
        ttk.Label(angular_row, textvariable=self._angular_step_size, width=6,
                  style="Value.TLabel").pack(side="right")
        self._build_jog_column(panel, "ANGULAR", (
            ("+ROLL", "a+"), ("-ROLL", "a-"), ("+PITCH", "b+"),
            ("-PITCH", "b-"), ("+YAW", "c+"), ("-YAW", "c-")), self._angular_step_size)

    def _build_jog_column(self, panel, title, buttons, step_variable):
        ttk.Label(panel, text=title, style="PanelText.TLabel").pack(anchor="w", pady=(0, 5))
        column = ttk.Frame(panel, style="Panel.TFrame")
        column.pack(fill="x")
        for label, key in buttons:
            button = ttk.Button(column, text=label, style="CartAction.TButton", width=7)
            button.pack(fill="x", pady=2)
            button.bind("<ButtonPress-1>", lambda _event, jog=key: self._start_jog(jog))
            button.bind("<ButtonRelease-1>", lambda _event, jog=key: self._stop_jog(jog))

        for child in column.winfo_children():
            child.pack_configure(padx=5, ipadx=4, ipady=4)

    def _load_poses(self):
        try:
            with self._pose_file.open("r", encoding="utf-8") as pose_file:
                poses = json.load(pose_file)
            return poses if isinstance(poses, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write_poses(self):
        self._pose_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix="joint_configurations.", suffix=".tmp", dir=self._pose_file.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as pose_file:
                descriptor = None
                json.dump(self._poses, pose_file, indent=2)
                pose_file.write("\n")
                pose_file.flush()
                os.fsync(pose_file.fileno())
            os.replace(temporary_path, self._pose_file)
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def _refresh_pose_dropdown(self):
        names = sorted(self._poses.get(self._robot_id.get(), {}).keys())
        self._pose_dropdown["values"] = names
        if self._selected_pose.get() not in names:
            self._selected_pose.set(names[0] if names else "")

    def _save_pose(self):
        pose_name = self._pose_name.get().strip()
        robot_id = self._robot_id.get().strip().strip("/")
        if not pose_name:
            messagebox.showerror("Missing pose name", "Enter a name before saving the joint configuration.")
            return
        if not robot_id:
            messagebox.showerror("Missing robot ID", "Connect to a robot ID before saving a pose.")
            return
        self._poses.setdefault(robot_id, {})[pose_name] = [
            round(float(value.get()), 6) for value in self._joint_values
        ]
        try:
            self._write_poses()
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not write the pose file:\n{exc}")
            return
        self._refresh_pose_dropdown()
        self._selected_pose.set(pose_name)
        self._status.set(f"Saved pose '{pose_name}' for /armos/{robot_id}")

    def _move_to_pose(self):
        pose_name = self._selected_pose.get()
        robot_poses = self._poses.get(self._robot_id.get(), {})
        joint_values = robot_poses.get(pose_name)
        if joint_values is None:
            messagebox.showerror("No pose selected", "Choose a saved joint configuration first.")
            return
        if not self._enabled.get():
            messagebox.showwarning("UI inactive", "Set UI ACTIVE before moving to a saved pose.")
            return
        for variable, joint_value in zip(self._joint_values, joint_values):
            variable.set(joint_value)
        self._publish_joint()
        self._status.set(f"Moving /armos/{self._robot_id.get()} to '{pose_name}'")

    def _on_telemetry(self, message: ArmosTelemetry):
        row = {
            "timestamp": float(message.timestamp),
            "joint_pos": [float(value) for value in message.joint_pos],
            "joint_vel": [float(value) for value in message.joint_vel],
            "joint_torque": [float(value) for value in message.joint_torque],
            "wrench": [float(value) for value in message.wrench],
            "tcp_pose": [float(value) for value in message.tcp_pose],
            "tcp_twist": [float(value) for value in message.tcp_twist],
            "manipulability": float(message.manipulability),
            "singularity_distance": float(message.singularity_distance),
        }
        with self._telemetry_lock:
            self._latest_telemetry = row
            if self._logging_active:
                self._telemetry_rows.append(row)

    def _refresh_joint_positions(self):
        with self._telemetry_lock:
            telemetry = self._latest_telemetry
        if telemetry is None:
            messagebox.showwarning("No telemetry", "No telemetry has been received from this robot yet.")
            return
        joint_positions = telemetry["joint_pos"]
        if len(joint_positions) < len(self._joint_values):
            messagebox.showwarning(
                "Incomplete telemetry",
                f"Received {len(joint_positions)} joint positions; expected 7.",
            )
        for variable, joint_value in zip(self._joint_values, joint_positions):
            variable.set(joint_value)
        self._status.set(f"Joint positions refreshed from /armos/{self._robot_id.get()}")

    def _toggle_logging(self):
        if self._logging_active:
            self._stop_logging()
            return
        with self._telemetry_lock:
            self._telemetry_rows = []
            self._logging_active = True
        self._logging.set(True)
        self._log_button_text.set("LOG ON")
        self._log_button.configure(style="LogOn.TButton")
        self._status.set(f"Logging telemetry from /armos/{self._robot_id.get()}")

    def _stop_logging(self):
        with self._telemetry_lock:
            self._logging_active = False
            rows = self._telemetry_rows
            self._telemetry_rows = []
        self._logging.set(False)
        self._log_button_text.set("LOG OFF")
        self._log_button.configure(style="LogOff.TButton")
        if not rows:
            self._status.set("Logging stopped; no telemetry samples collected")
            return

        robot_id = self._robot_id.get().strip().strip("/") or "unknown"
        safe_robot_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in robot_id)
        log_directory = Path.home() / ".config" / "armos" / "telemetry"
        log_directory.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_robot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        log_path = log_directory / filename
        fields = (
            "timestamp", "joint_pos", "joint_vel", "joint_torque", "wrench",
            "tcp_pose", "tcp_twist", "manipulability", "singularity_distance",
        )
        try:
            with log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(log_file, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        field: json.dumps(row[field], separators=(",", ":"))
                        if isinstance(row[field], list) else row[field]
                        for field in fields
                    })
        except OSError as exc:
            messagebox.showerror("Log save failed", f"Could not save telemetry CSV:\n{exc}")
            return
        self._status.set(f"Saved {len(rows)} telemetry samples to {log_path}")

    def _set_robot_id(self):
        robot_id = self._robot_id.get().strip().strip("/")
        if not robot_id or any(char.isspace() for char in robot_id):
            messagebox.showerror("Invalid robot ID", "Enter a non-empty robot ID without spaces.")
            return
        self._stop_robot()
        if self._logging_active:
            self._stop_logging()
        if self._telemetry_sub is not None:
            self.destroy_subscription(self._telemetry_sub)
            self._telemetry_sub = None
        self._robot_id.set(robot_id)
        self._joint_pub = self.create_publisher(ArmosJointControl, f"/armos/{robot_id}/joint_control", 10)
        self._cart_pub = self.create_publisher(ArmosCartControl, f"/armos/{robot_id}/cart_control", 10)
        self._telemetry_sub = self.create_subscription(
            ArmosTelemetry,
            f"/armos/{robot_id}/telemetry",
            self._on_telemetry,
            10,
        )
        self._refresh_pose_dropdown()
        state = "active" if self._enabled.get() else "inactive"
        self._status.set(f"Connected to /armos/{robot_id} | UI {state}")

    def _toggle_enabled(self):
        self._enabled.set(not self._enabled.get())
        if self._enabled.get():
            self._status.set(f"Publishing commands to /armos/{self._robot_id.get()}")
            self._ui_active_text.set("UI ACTIVE")
            self._active_button.configure(style="UiActive.TButton")
            self._publish_joint()
        else:
            self._jogging.clear()
            self._status.set("Output disabled")
            self._ui_active_text.set("UI INACTIVE")
            self._active_button.configure(style="UiInactive.TButton")
            self._stop_robot()

    def _publish_joint(self, force=False):
        if (not self._enabled.get() and not force) or self._joint_pub is None:
            return
        message = ArmosJointControl()
        if force:
            message.mode = 1
            joint_values = (0.0,) * 7
        else:
            message.mode = 0
            joint_values = (float(value.get()) for value in self._joint_values)
        for name, value in zip(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"), joint_values):
            setattr(message, name, value)
        self._joint_pub.publish(message)

    def _start_jog(self, direction):
        if not self._enabled.get() or direction in self._jogging:
            return
        self._jogging[direction] = True
        self._jog_once(direction)

    def _jog_once(self, direction):
        if direction not in self._jogging:
            return
        direction_vector = {
            "x+": ((1, 0, 0), self._step_size), "x-": ((-1, 0, 0), self._step_size),
            "y+": ((0, 1, 0), self._step_size), "y-": ((0, -1, 0), self._step_size),
            "z+": ((0, 0, 1), self._step_size), "z-": ((0, 0, -1), self._step_size),
            "a+": ((0, 0, 0, 1, 0, 0), self._angular_step_size),
            "a-": ((0, 0, 0, -1, 0, 0), self._angular_step_size),
            "b+": ((0, 0, 0, 0, 1, 0), self._angular_step_size),
            "b-": ((0, 0, 0, 0, -1, 0), self._angular_step_size),
            "c+": ((0, 0, 0, 0, 0, 1), self._angular_step_size),
            "c-": ((0, 0, 0, 0, 0, -1), self._angular_step_size),
        }[direction]
        vector, step_variable = direction_vector
        if len(vector) == 3:
            velocity = tuple(axis * step_variable.get() for axis in vector) + (0.0, 0.0, 0.0)
        else:
            velocity = tuple(axis * step_variable.get() for axis in vector)
        self._publish_cart_velocity(velocity)
        self.root.after(80, lambda: self._jog_once(direction))

    def _stop_jog(self, direction):
        self._jogging.pop(direction, None)
        self._publish_cart_velocity((0.0,) * 6)

    def _publish_cart_velocity(self, velocity, force=False):
        if (not self._enabled.get() and not force) or self._cart_pub is None:
            return
        message = ArmosCartControl()
        message.mode = 1
        message.x, message.y, message.z, message.a, message.b, message.c = velocity
        self._cart_pub.publish(message)

    def _stop_robot(self):
        self._jogging.clear()
        self._publish_joint(force=True)
        self._publish_cart_velocity((0.0,) * 6, force=True)
        self._status.set("STOP command sent")

    def _close(self):
        self._stop_robot()
        if self._logging_active:
            self._stop_logging()
        if self._telemetry_sub is not None:
            self.destroy_subscription(self._telemetry_sub)
            self._telemetry_sub = None
        self.destroy_node()
        rclpy.shutdown()
        self.root.destroy()


def main():
    rclpy.init()
    root = tk.Tk()
    node = ArmosControlGui(root)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        root.mainloop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()