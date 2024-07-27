import sys
import csv
import wolframclient
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re

class ATLAS:
    def __init__(self, input_file):
        self.input_file = input_file
        self.simulation_params = {}
        self.physical_constants = {}
        self.output_settings = {}
        self.bodies = []
        self.additional_physics = {}

    def read_input(self):
        with open(self.input_file, 'r') as file:
            current_section = None
            current_body = None

            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if ':' in line:
                    key, value = map(str.strip, line.split(':', 1))
                    
                    if key == 'NUM_BODIES':
                        current_section = 'bodies'
                        self.simulation_params[key] = int(value)
                    elif key in ['MASS', 'POSITION', 'VELOCITY', 'RADIUS', 'NAME']:
                        if current_body is None:
                            current_body = {}
                        current_body[key.lower()] = self.parse_value(value)
                        if len(current_body) == 5:  # All body parameters are set
                            self.bodies.append(current_body)
                            current_body = None
                    elif current_section == 'bodies':
                        continue  # Skip body number lines
                    elif key in ['TOTAL_TIME', 'TIME_STEP', 'INTEGRATION_METHOD']:
                        self.simulation_params[key.lower()] = self.parse_value(value)
                    elif key == 'G':
                        self.physical_constants[key] = self.parse_value(value)
                    elif key in ['OUTPUT_INTERVAL', 'CSV_FILENAME', 'GIF_FILENAME', 'GIF_SPEED']:
                        self.output_settings[key.lower()] = self.parse_value(value)
                    elif key in ['RELATIVISTIC_EFFECTS', 'COLLISION_DETECTION']:
                        self.additional_physics[key.lower()] = self.parse_value(value)

        print("Input file parsed successfully.")
        self.print_parsed_data()

    def parse_value(self, value):
        # Try to convert to float if possible
        try:
            return float(value)
        except ValueError:
            pass
        
        # Check for boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Check for vector values
        if ',' in value:
            return [float(x.strip()) for x in value.split(',')]
        
        # Return as string if all else fails
        return value

    def print_parsed_data(self):
        print("\nSimulation Parameters:")
        print(self.simulation_params)
        print("\nPhysical Constants:")
        print(self.physical_constants)
        print("\nOutput Settings:")
        print(self.output_settings)
        print("\nCelestial Bodies:")
        for body in self.bodies:
            print(body)
        print("\nAdditional Physics:")
        print(self.additional_physics)

    def calculate_trajectories(self):
        session = WolframLanguageSession()
        try:
            # Wolfram Language 코드 생성
            code = self.generate_wolfram_code()
            
            # Wolfram Language 코드 실행
            result = session.evaluate(wlexpr(code))
            
            # 결과 처리
            trajectories = self.process_wolfram_result(result)
            
            return trajectories
        
        finally:
            session.terminate()

    def generate_wolfram_code(self):
        # Wolfram Language 코드 생성
        code = f"""
        G = {self.physical_constants['G']};
        bodies = {self.bodies};
        tmax = {self.simulation_params['total_time']};
        dt = {self.simulation_params['time_step']};
        
        n = Length[bodies];
        m = Table[bodies[[i, "mass"]], {{i, n}}];
        
        initialConditions = Flatten[Table[
            {{
                x[i][0] == bodies[[i, "position", 1]],
                y[i][0] == bodies[[i, "position", 2]],
                z[i][0] == bodies[[i, "position", 3]],
                x'[i][0] == bodies[[i, "velocity", 1]],
                y'[i][0] == bodies[[i, "velocity", 2]],
                z'[i][0] == bodies[[i, "velocity", 3]]
            }},
            {{i, n}}
        ]];
        
        equations = Flatten[Table[
            {{
                x''[i][t] == Sum[G m[[j]] (x[j][t] - x[i][t])/((x[j][t] - x[i][t])^2 + (y[j][t] - y[i][t])^2 + (z[j][t] - z[i][t])^2)^(3/2), {{j, n, j != i}}],
                y''[i][t] == Sum[G m[[j]] (y[j][t] - y[i][t])/((x[j][t] - x[i][t])^2 + (y[j][t] - y[i][t])^2 + (z[j][t] - z[i][t])^2)^(3/2), {{j, n, j != i}}],
                z''[i][t] == Sum[G m[[j]] (z[j][t] - z[i][t])/((x[j][t] - x[i][t])^2 + (y[j][t] - y[i][t])^2 + (z[j][t] - z[i][t])^2)^(3/2), {{j, n, j != i}}]
            }},
            {{i, n}}
        ]];
        
        sol = NDSolve[Join[equations, initialConditions], 
                      Flatten[Table[{{x[i], y[i], z[i]}}, {{i, n}}]], 
                      {{t, 0, tmax}}, 
                      MaxSteps -> Infinity];
        
        Table[{{x[i][t], y[i][t], z[i][t]}} /. sol, {{i, n}}]
        """
        return code

    def process_wolfram_result(self, result):
        # Wolfram 결과를 Python 데이터 구조로 변환
        trajectories = []
        for body_result in result:
            body_trajectory = []
            for point in body_result[0]:
                x, y, z = point
                body_trajectory.append([float(x), float(y), float(z)])
            trajectories.append(np.array(body_trajectory))
        return trajectories

    def save_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 헤더 작성
            header = ['Time']
            for i, body in enumerate(self.bodies):
                header.extend([f"Body{i+1}_X", f"Body{i+1}_Y", f"Body{i+1}_Z"])
            writer.writerow(header)
            
            # 데이터 작성
            num_timesteps = len(self.results[0])
            for t in range(num_timesteps):
                row = [t * self.simulation_params['time_step']]  # 시간 추가
                for body_trajectory in self.results:
                    row.extend(body_trajectory[t])  # 각 천체의 x, y, z 좌표 추가
                writer.writerow(row)

        print(f"CSV file saved: {filename}")


    def update_animation(self, frame, trails, scatters):
        for body_index, body_trajectory in enumerate(self.results):
            # 궤적 업데이트
            trails[body_index].set_data(body_trajectory[:frame, 0], body_trajectory[:frame, 1])
            trails[body_index].set_3d_properties(body_trajectory[:frame, 2])
            
            # 현재 위치 업데이트
            scatters[body_index]._offsets3d = (
                body_trajectory[frame:frame+1, 0],
                body_trajectory[frame:frame+1, 1],
                body_trajectory[frame:frame+1, 2]
            )
        
        return trails + scatters

    def create_animation(self, filename):
        # 3D 그래프 설정
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 각 천체의 궤적을 저장할 리스트
        trails = []
        # 각 천체를 나타내는 산점도를 저장할 리스트
        scatters = []

        # 축 범위 계산
        all_positions = np.concatenate(self.results)
        max_range = np.max(all_positions.max(axis=0) - all_positions.min(axis=0)) / 2.0
        mid_x = (all_positions[:, 0].min() + all_positions[:, 0].max()) / 2
        mid_y = (all_positions[:, 1].min() + all_positions[:, 1].max()) / 2
        mid_z = (all_positions[:, 2].min() + all_positions[:, 2].max()) / 2

        # 각 천체에 대한 초기 설정
        colors = plt.cm.jet(np.linspace(0, 1, len(self.results)))
        for body_index, body_trajectory in enumerate(self.results):
            trail, = ax.plot([], [], [], color=colors[body_index], linewidth=1, alpha=0.5)
            scatter = ax.scatter([], [], [], color=colors[body_index], s=50)
            trails.append(trail)
            scatters.append(scatter)

        # 축 레이블 및 제목 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('N-Body Gravitational Simulation')

        # 축 범위 설정
        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

        # 애니메이션 생성
        anim = animation.FuncAnimation(fig, 
                                       lambda frame: self.update_animation(frame, trails, scatters), 
                                       frames=len(self.results[0]),
                                       interval=50, blit=False)

        # GIF로 저장
        anim.save(filename, writer='pillow', fps=30)
        plt.close(fig)

        print(f"Animation saved as {filename}")

    def run(self):
        self.read_input()
        self.results = self.calculate_trajectories()
        csv_filename = self.output_settings['csv_filename']
        self.save_csv(csv_filename)
        gif_filename = self.output_settings['gif_filename']
        self.create_animation(gif_filename)

def main(input_file):
    atlas = ATLAS(input_file)
    atlas.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python atlas.py input_file.inp")
        sys.exit(1)
    main(sys.argv[1])