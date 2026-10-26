"""Entry point: rclpy executor in a background thread, Qt in the main thread."""

import os
import signal
import sys
import threading
import traceback

import rclpy
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from . import theme
from .main_window import MainWindow
from .ros_bridge import RosBridge
from .state import RobotState


def main():
    # PyQt5 aborts the process on an exception escaping a slot; print it and keep running
    # so the E-STOP button stays available.
    sys.excepthook = lambda *exc: traceback.print_exception(*exc)

    # Own SIGINT/SIGTERM from the very first instruction. rclpy's default handlers would only
    # shut down the ROS context and leave the Qt loop running, so `ros2 launch` could not stop
    # the UI when it was signalled during startup.
    stop = threading.Event()
    app = None

    def on_signal(*_):
        stop.set()
        if app is not None:
            app.quit()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    rclpy.init(args=sys.argv, signal_handler_options=SignalHandlerOptions.NO)
    bridge = executor = None
    rc = 0
    try:
        app = QApplication(remove_ros_args(sys.argv))
        app.setApplicationName('Franka Control Deck')
        app.setStyleSheet(theme.build_qss())

        bridge = RosBridge(state=None)
        state = RobotState(buffer_seconds=bridge.cfg['plot.buffer_seconds'],
                           sample_hz=bridge.cfg['plot.sample_hz'],
                           n_joints=len(bridge.joint_names))
        bridge.state = state

        executor = SingleThreadedExecutor()
        executor.add_node(bridge)
        threading.Thread(target=executor.spin, daemon=True).start()

        window = MainWindow(bridge, state)
        window.show()

        # Qt blocks Python signal handlers while idle; a periodic no-op lets them run.
        wake = QTimer()
        wake.timeout.connect(lambda: None)
        wake.start(200)

        if not stop.is_set():   # a signal before exec_() would otherwise be lost
            rc = app.exec_()
    except Exception:  # noqa: BLE001 - report startup failures, then still exit below
        traceback.print_exc()
        rc = 1
    finally:
        for cleanup in (lambda: bridge and bridge.disable(),
                        lambda: executor and executor.shutdown(timeout_sec=1.0),
                        lambda: bridge and bridge.destroy_node(),
                        rclpy.try_shutdown):
            try:
                cleanup()
            except Exception:  # noqa: BLE001 - best effort; must not block exit
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        # Hard exit: Qt/DDS teardown at interpreter exit can hang and leave launch waiting.
        os._exit(rc)


if __name__ == '__main__':
    main()
