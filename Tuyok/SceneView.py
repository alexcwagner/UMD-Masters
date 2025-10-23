
import sys
#import struct
import numpy as np

from Shaders import Shader
from Mesh import Mesh
from Vao import Vao

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QSurfaceFormat, QPainter, QColor, QFont
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL import GL
#import OpenGL.GL.shaders
from OpenGL.GL import shaders


class SceneView(QOpenGLWidget):
    def __init__(self, scene=None, parent=None):
        super().__init__(parent)
        
        # Camera (Z up, Y forward, X right; RH system)
        self.angle_x = 20.0
        self.angle_y = -30.0
        self.distance = 3.0

        # Render/state toggles
        self.wireframe = False
        self.cull = True
        self.front_cw = False  # False=CCW (default), True=CW
        self.auto_rotate = True

        self.last_pos = None

        self.setFocusPolicy(Qt.StrongFocus)  # <-- ensure widget can take keyboard focus

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_angle)
        self.timer.start(16)

    def initializeGL(self):
        GL.glClearDepth(1.0)

        self.shader = Shader()
        self.mesh = Mesh.test_cube(flat=True)      
        self.vao = Vao(self.shader, self.mesh)


    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        # --- Reassert GL state EACH FRAME (QPainter may change it) ---
        try:
            GL.glDisable(GL.GL_BLEND)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDepthFunc(GL.GL_LESS)
            if self.cull:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glCullFace(GL.GL_BACK)
            else:
                GL.glDisable(GL.GL_CULL_FACE)
            GL.glFrontFace(GL.GL_CW if self.front_cw else GL.GL_CCW)    
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    
            # --- 3D draw ---
            #GL.glUseProgram(self.shader.program)
            #GL.glBindVertexArray(self.vao._vao)
    
            aspect = self.width() / max(1, self.height())
            proj = self.perspective(45, aspect, 0.1, 100.0)
    
            eye = self.spherical_to_cartesian(self.distance,
                                              np.radians(self.angle_x),
                                              np.radians(self.angle_y))
            view = self.lookAt(eye, np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
            
            model = np.diag([0.5, 0.5, 0.5, 1.0]).astype(np.float32)
            
            self.shader.model = model
            self.shader.view = view
            self.shader.projection = proj
    
            #GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if self.wireframe else GL.GL_FILL)
            #GL.glDrawElements(GL.GL_TRIANGLES, len(self.vao.indices), GL.GL_UNSIGNED_INT, None)
            self.vao.draw()

    
            # --- 2D overlay ---
            self.draw_axis_overlay()
        except Exception as e:
            print("Fatal error in paintGL:", e, file=sys.stderr)
            import traceback; traceback.print_exc()
            QApplication.quit()     # clean shutdown
            sys.exit(1)             # forcefully terminate

    def draw_axis_overlay(self):
        size = 50
        margin = 20
        origin_x = margin + size
        origin_y = self.height() - margin - size

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 10, QFont.Bold))

        painter.setPen(QColor(255, 0, 0))     # X
        painter.drawLine(origin_x, origin_y, origin_x + size, origin_y)
        painter.drawText(origin_x + size + 5, origin_y, "X")

        painter.setPen(QColor(0, 200, 0))     # Y
        painter.drawLine(origin_x, origin_y, origin_x, origin_y + size)
        painter.drawText(origin_x - 5, origin_y + size + 15, "Y")

        painter.setPen(QColor(0, 0, 255))     # Z
        painter.drawLine(origin_x, origin_y, origin_x, origin_y - size)
        painter.drawText(origin_x - 15, origin_y - size - 5, "Z")

        painter.setPen(QColor(255, 255, 255))
        hud = f"Cull: {'ON' if self.cull else 'OFF'} | Front: {'CW' if self.front_cw else 'CCW'}"
        painter.drawText(self.width() - 220, 25, hud)

        painter.end()

    def update_angle(self):
        if self.auto_rotate:
            self.angle_y += 0.5
        self.update()

    def keyPressEvent(self, event):
        ch = event.text().lower()
        if ch == 'w':
            self.wireframe = not self.wireframe
        elif ch == 'r':
            self.auto_rotate = not self.auto_rotate
        elif ch == 'k':
            self.cull = not self.cull
        elif ch == 'f':
            self.front_cw = not self.front_cw
        self.update()

    def mousePressEvent(self, event):
        self.setFocus()  # <-- grab focus on click
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        if event.buttons() & Qt.LeftButton:
            self.angle_x += dy * 0.5
            self.angle_y += dx * 0.5
            self.auto_rotate = False
        elif event.buttons() & Qt.RightButton:
            self.distance *= (1.0 - dy * 0.01)
        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.distance *= (1.0 - delta * 0.1)
        self.update()

    # --- Math helpers ---
    def spherical_to_cartesian(self, r, pitch, yaw):
        x = r * np.cos(pitch) * np.cos(yaw)
        y = r * np.cos(pitch) * np.sin(yaw)
        z = r * np.sin(pitch)
        return np.array([x, y, z], dtype=np.float32)

    def perspective(self, fovy, aspect, znear, zfar):
        f = 1.0 / np.tan(np.radians(fovy) / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (zfar+znear)/(znear-zfar), (2*zfar*znear)/(znear-zfar)],
            [0, 0, -1, 0],
        ], dtype=np.float32)

    def lookAt(self, eye, center, up):
        f = center - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)     # right
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)      # true up

        m = np.identity(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f

        t = np.identity(4, dtype=np.float32)
        t[:3, 3] = -eye
        return m @ t

if __name__ == "__main__":
    print("performing test")

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("STL Viewer")
            self.resize(800, 600)
            self.viewer = SceneView()
            self.setupUI()
        
        def setupUI(self):
            self.setCentralWidget(self.viewer)
            

    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
            
            
            