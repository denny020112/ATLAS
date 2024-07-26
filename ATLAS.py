import sys
import csv
import wolframclient
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


    def create_animation(self, filename):
        # 결과를 GIF 애니메이션으로 저장하는 메서드
        fig, ax = plt.subplots()
        # 여기에 애니메이션 생성 로직 구현
        anim = animation.FuncAnimation(fig, self.update_animation, frames=len(self.results), interval=50)
        anim.save(filename, writer='pillow')

    def update_animation(self, frame):
        # 애니메이션 프레임 업데이트 메서드
        # 여기에 프레임 업데이트 로직 구현
        pass

    def run(self):
        self.read_input()
        self.results = self.calculate_trajectories()
        self.save_csv(self.input_file.replace('.inp', '_output.csv'))
        self.create_animation(self.input_file.replace('.inp', '_animation.gif'))

def main(input_file):
    atlas = ATLAS(input_file)
    atlas.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python atlas.py input_file.inp")
        sys.exit(1)
    main(sys.argv[1])