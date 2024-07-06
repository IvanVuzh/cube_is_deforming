from math import sqrt, floor
import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# region terminal color definitions
CRED = '\033[91m'
CGREEN  = '\33[32m'
CYELLOW2 = '\33[93m'
CEND = '\033[0m'
# endregion

# region PointCalculator
class PointCalculator:
    def __init__(
        self, 
        ax: int,
        ay: int,
        az: int,
        nx: int,
        ny: int,
        nz: int):

        # a_ == відстань (довжина/ширина/глибина) на певній координаті
        # n_ == кількість сегментів поділених по певній координаті
        self.__ax: int = ax
        self.__ay: int = ay
        self.__az: int = az
        
        self.__nx: int = nx
        self.__ny: int = ny
        self.__nz: int = nz

        self.__step_x: float = ax / nx
        self.__step_y: float = ay / ny
        self.__step_z: float = az / nz

        # по кожному сегменту має бути почка посередині на кожному ребрі
        # тому кожен крок зменшуємо вдвічі
        self.short_x_step: float = self.__step_x / 2
        self.short_y_step: float = self.__step_y / 2
        self.short_z_step: float = self.__step_z / 2

        # self.__x_start_coord = -ax / 2
        # self.__y_start_coord = -ay / 2
        # self.__z_start_coord = -az / 2
        
        self.__x_start_coord = 0
        self.__y_start_coord = 0
        self.__z_start_coord = 0

        # self.__x_end_coord = ax / 2
        # self.__y_end_coord = ay / 2
        # self.__z_end_coord = az / 2
        
        self.__x_end_coord = ax
        self.__y_end_coord = ay
        self.__z_end_coord = az

        self.__akt: list[list[float, float, float]] # тут не tuple, бо 'tuple' object does not support item assignment
        self.__nt: list[tuple[float, float, float]]
        self.__zu: list[tuple[float, float, float]]
        self.__zp = []
        self.__DFIABG = list[tuple[int, int, float]]
        self.__DJ_matrixes = []
        self.__DJ = []
        self.__DFIXYZ = []
        self.__MGE = []
        self.__FE = []
        self.__MG_GENERAL = []
        self.__F_GENERAL = []
        self.__Deformation_Info = []
        
        self.__faces = ["top", "bottom", "left", "right", "front", "back"]
        self.__gauss_points = [-sqrt(0.6), 0, sqrt(0.6)]
        self.__gauss_constant = [5/9, 8/9, 5/9]
        # модуль Юнга
        # для легкості берем 1
        self.__E = 1
        # коефіцієнт Пуассона 0.0 - 0.5 (візьму для заліза)
        self.__v = 0.3
        
        self.__lambda = self.__E / ((1 + self.__v) * (1 - 2 * self.__v))
        self.__mu = self.__E / ((1 + self.__v) * 2)
        
        self.AKT()
        self.NT()
        self.ZP()
        self.ZU()
        
    @staticmethod
    def get_nt_point_for_face(face: str):
        """return array of points for specified face in LOCAL FINITE ELEMENT scheme

        Args:
            face (str): face name

        Returns:
            _type_: integer array
        """
        if "top":
            return [4, 5, 6, 7, 16, 17, 18, 19]
        if "bottom":
            return [0, 1, 2, 3, 8, 9, 10, 11]
        if "left":
            return [3, 0, 4, 7, 11, 12, 19, 15]
        if "right":
            return [1, 2, 6, 5, 9, 14, 17, 13]
        if "front":
            return [0, 1, 5, 4, 8, 13, 16, 12]
        if "back":
            return [2, 3, 7, 6, 10, 15, 18, 14]

    def AKT(self):
        # Generate coordinates (akt)
        # рахуємо по прикладу як на парі  
        # знизу зліва направо по Х найближчі до нас (цикл по Х)
        # далі перескакуємо "на рядок назад", тобто наступний по Y рядок Х (цикл по Y)
        # далі йдемо "вгору" (цикл по Z)
        # перша пара запис 11 групи 7:00
        akt = []
        for z in range(0, 2 * (self.__nz) + 1):
            for y in range(0, 2 * (self.__ny) + 1):
                for x in range(0, 2 * (self.__nx) + 1):
                    x_coord: float = x * self.short_x_step
                    y_coord: float = y * self.short_y_step
                    z_coord: float = z * self.short_z_step

                    on_half_x: bool = abs(x_coord - self.__x_start_coord) % self.__step_x == self.short_x_step
                    on_half_y: bool = abs(y_coord - self.__y_start_coord) % self.__step_y == self.short_y_step
                    on_half_z: bool = abs(z_coord - self.__z_start_coord) % self.__step_z == self.short_z_step
                    # не додаємо точки, які будуть серединами мінімум двох сторін
                    if not ((on_half_x and on_half_y) or (on_half_x and on_half_z) or (on_half_y and on_half_z)):
                        akt.append([x_coord, y_coord, z_coord])

        self.__akt = akt
        self.ZU()

        return akt

    def NT(self):
        # спочатку кутові вузли
        # стартуємо низ ліво і йдемо проти годинникої стрілки якщо дивитись зверху
        # далі верх ліво і так само
        # потім беремо середини 
        # починаємо низ між 1 і 2 -||-
        # далі вертикальні ребра куба (починаємо між 1 і 5)
        # далі верх так як робили низ
        nt: list[list[int]] = []

        # print(self.__step_x)
        # print(self.__step_y)
        # print(self.__step_z)

        # звичайний Range не працює, бо тут флоати (люблю пайтон)
        for z in np.arange(self.__z_start_coord, self.__z_end_coord, self.__step_z):
            for y in np.arange(self.__y_start_coord, self.__y_end_coord, self.__step_y):
                for x in np.arange(self.__x_start_coord, self.__x_end_coord, self.__step_x):
                    nt.append([
                        # "основа" кути
                        1 + self.__akt.index([x, y, z]),                                                        # 1
                        1 + self.__akt.index([x + self.__step_x, y, z]),                                        # 2
                        1 + self.__akt.index([x + self.__step_x, y + self.__step_y, z]),                        # 3
                        1 + self.__akt.index([x, y + self.__step_y, z]),                                        # 4
                        # "верхня сторона" кути
                        1 + self.__akt.index([x, y, z + self.__step_z]),                                        # 5
                        1 + self.__akt.index([x + self.__step_x, y, z + self.__step_z]),                        # 6
                        1 + self.__akt.index([x + self.__step_x, y + self.__step_y, z + self.__step_z]),        # 7
                        1 + self.__akt.index([x, y + self.__step_y, z + self.__step_z]),                        # 8
                        # основа між кутами
                        1 + self.__akt.index([x + self.short_x_step, y, z]),                                    # 9
                        1 + self.__akt.index([x + self.__step_x, y + self.short_y_step, z]),                    # 10
                        1 + self.__akt.index([x + self.short_x_step, y + self.__step_y, z]),                    # 11
                        1 + self.__akt.index([x, y + self.short_y_step, z]),                                    # 12
                        # середини вертикальних ребер
                        1 + self.__akt.index([x, y, z + self.short_z_step]),                                    # 13
                        1 + self.__akt.index([x + self.__step_x, y, z + self.short_z_step]),                    # 14
                        1 + self.__akt.index([x + self.__step_x, y + self.__step_y, z + self.short_z_step]),    # 15
                        1 + self.__akt.index([x, y + self.__step_y, z + self.short_z_step]),                    # 16
                        # "стеля" між кутами
                        1 + self.__akt.index([x + self.short_x_step, y, z + self.__step_z]),                    # 17
                        1 + self.__akt.index([x + self.__step_x, y + self.short_y_step, z + self.__step_z]),    # 18
                        1 + self.__akt.index([x + self.short_x_step, y + self.__step_y, z + self.__step_z]),    # 19
                        1 + self.__akt.index([x, y + self.short_y_step, z + self.__step_z]),                    # 20
                    ])

        self.__nt = nt

        return nt

    # задається юзером
    def ZP(self):
        # [номер вузла, сторона, сила]
        # можна вернути захардкодженний аррей
        # зараз це давлять на верх на всіх верхніх точках
        zp=[[] for _ in range(self.__nt.__len__())]
        for nt_index, nt_i in enumerate(self.__nt):
            top_points_indexes = [point for point in nt_i if self.__akt[point - 1][2] == self.__z_end_coord]
            zp[nt_index] = [[point_index, self.__akt[point_index - 1], self.__faces[0], 0.0005] for point_index in top_points_indexes]
        
        # for nt_index, nt_i in enumerate(self.__nt):
        #     top_points_indexes = [point for point in nt_i if self.__akt[point - 1][2] == self.__z_end_coord]
        #     back_points_indexes = [point for point in nt_i if self.__akt[point - 1][0] == self.__x_start_coord]
        #     top_points_data = [[point_index, self.__akt[point_index - 1], self.__faces[0], 0.05] for point_index in top_points_indexes]
        #     back_points_data = [[point_index, self.__akt[point_index - 1], self.__faces[4], 0.2 ] for point_index in back_points_indexes]
        #     # print('top', top_points_data)
        #     # print('right', right_points_data)
        #     # print('all', top_points_data +right_points_data)
        #     if(top_points_data.__len__() > 0 or back_points_data.__len__() > 0):
        #         zp[nt_index] = top_points_data + back_points_data

        # for nt_index, nt_i in enumerate(self.__nt):
        #     top_points_indexes = [point for point in nt_i if self.__akt[point - 1][0] == self.__x_end_coord]
        #     zp[nt_index] = [[point_index, self.__akt[point_index - 1], self.__faces[2], 0.05] for point_index in top_points_indexes]
        

        self.__zp = zp

        return zp

    # ZU - масив елементів, які не рухатимуться (в нашому  випадку просто основа, бо вона "приварена")
    def ZU(self):
        zu = [index + 1 for index, item in enumerate(self.__akt) if item[0] == self.__x_start_coord]
        self.__zu = zu
        return zu

    # тут похідні по тих функціях шо в лекції
    # для точок 1-8
    # checked
    @staticmethod
    def dFI_to_dAlpha_1_8(alpha, alphai, beta, betai, gamma, gammai):
        return 0.125 * (1 + beta * betai) * (1 + gamma * gammai) \
            * (alphai * (-2 + alpha * alphai + gamma * gammai + beta * betai) + alphai * (1 + alpha * alphai))
    
    @staticmethod
    def dFI_to_dBeta_1_8(alpha, alphai, beta, betai, gamma, gammai):
        return 0.125 * (1 + alpha * alphai) * (1 + gamma * gammai) \
            * (betai * (-2 + alpha * alphai + gamma * gammai + beta * betai) + betai * (1 + beta * betai))

    @staticmethod
    def dFI_to_dGamma_1_8(alpha, alphai, beta, betai, gamma, gammai):
        return 0.125 * (1 + beta * betai) * (1 + alpha * alphai) \
            * (gammai * (-2 + alpha * alphai + gamma * gammai + beta * betai) + gammai * (1 + gamma * gammai))

    # для точок 9-20
    @staticmethod
    def dFI_to_dAlpha_9_20(alpha, alphai, beta, betai, gamma, gammai):
        return 0.25 * (1 + beta * betai) * (1 + gamma * gammai) \
            * (alphai * (
                    - betai * betai * gammai * gammai * alpha * alpha
                    - beta * beta * gammai * gammai * alphai * alphai
                    - betai * betai * gamma * gamma * alphai * alphai + 1) -
                (2 * betai * betai * gammai * gammai * alpha) * (alpha * alphai + 1))

    @staticmethod
    def dFI_to_dBeta_9_20(alpha, alphai, beta, betai, gamma, gammai):
        return 0.25 * (1 + alpha * alphai) * (1 + gamma * gammai) \
            * (betai * (
                    - betai * betai * gammai * gammai * alpha * alpha
                    - beta * beta * gammai * gammai * alphai * alphai
                    - betai * betai * gamma * gamma * alphai * alphai + 1) -
                (2 * beta * gammai * gammai * alphai * alphai) * (betai * beta + 1))
    
    @staticmethod
    def dFI_to_dGamma_9_20(alpha, alphai, beta, betai, gamma, gammai):
        return 0.25 * (1 + beta * betai) * (1 + alpha * alphai) \
            * (gammai * (
                    - betai * betai * gammai * gammai * alpha * alpha
                    - beta * beta * gammai * gammai * alphai * alphai
                    - betai * betai * gamma * gamma * alphai * alphai + 1) -
                (2 * betai * betai * gamma * alphai * alphai) * (gamma * gammai + 1))

    @staticmethod
    def createDFIABGKey(a, b, g, direction, i):
        key = f"{a}, {b}, {g}, {direction}, {i + 1}"
        return key

    # це рахується правильно, можна не перевіряти
    def DFIABG(self, willPrint: bool = False):
        coordinates = [
            [-1, 1, -1],    # 1
            [1, 1, -1],     # 2
            [1, -1, -1],    # 3
            [-1, -1, -1],   # 4
            
            [-1, 1, 1],     # 5
            [1, 1, 1],      # 6
            [1, -1, 1],     # 7
            [-1, -1, 1],    # 8
            
            [0, 1, -1],     # 9
            [1, 0, -1],     # 10
            [0, -1, -1],    # 11
            [-1, 0, -1],    # 12
            
            [-1, 1, 0],     # 13
            [1, 1, 0],      # 14
            [1, -1, 0],     # 15
            [-1, -1, 0],    # 16
            
            [0, 1, 1],      # 17
            [1, 0, 1],      # 18
            [0, -1, 1],     # 19
            [-1, 0, 1]      # 20
        ]

        # dFi по альфа бета гамма
        DFIABG = [[[0.0 for _ in range(3)] for _ in range(20)] for _ in range(27)]
        DFIABG_dict = {}

        directions = ["alpha", "beta", "gamma"]

        firstIndex = 0

        for gamma in range(3):
            for beta in range(3):
                for alpha in range(3):
                    for i in range(20):
                        for direction in directions:
                            alpha_value = self.__gauss_points[alpha] #-0.6
                            beta_value = self.__gauss_points[beta] #-0.6
                            gamma_value = self.__gauss_points[gamma] #-0.6

                            alpha_coordinate = coordinates[i][0]
                            beta_coordinate = coordinates[i][1]
                            gamma_coordinate = coordinates[i][2]

                            if (direction == "alpha" and i < 8):
                                derivative_value = self.dFI_to_dAlpha_1_8(alpha_value, alpha_coordinate, beta_value, beta_coordinate, gamma_value, gamma_coordinate)
                            elif (direction == "beta" and i < 8):
                                derivative_value = self.dFI_to_dBeta_1_8(alpha_value, alpha_coordinate, beta_value, beta_coordinate, gamma_value, gamma_coordinate)
                            elif (direction == "gamma" and i < 8):
                                derivative_value = self.dFI_to_dGamma_1_8(alpha_value, alpha_coordinate, beta_value, beta_coordinate, gamma_value, gamma_coordinate)
                            elif (direction == "alpha"):
                                derivative_value = self.dFI_to_dAlpha_9_20(alpha_value, alpha_coordinate, beta_value, beta_coordinate, gamma_value, gamma_coordinate)
                            elif (direction == "beta"):
                                derivative_value = self.dFI_to_dBeta_9_20(alpha_value, alpha_coordinate, beta_value, beta_coordinate, gamma_value, gamma_coordinate)
                            elif (direction == "gamma"):
                                derivative_value = self.dFI_to_dGamma_9_20(alpha_value, alpha_coordinate, beta_value, beta_coordinate, gamma_value, gamma_coordinate)
                            else:
                                raise ValueError('A very specific bad thing happened.')
                            
                            DFIABG[firstIndex][i][directions.index(direction)] = derivative_value

                            key = self.createDFIABGKey(alpha_value, beta_value, gamma_value, direction, i)
                            DFIABG_dict[key] = derivative_value

                    firstIndex = firstIndex + 1
        if willPrint: 
            for _key, _value in DFIABG_dict.items():
                print(f"{_key}: {_value}")

        self.__DFIABG = DFIABG

        return DFIABG

    # матриці для якобіанту переходу
    def DJ_matrixes(self):
        dj_matrixes = []
        
        self.DFIABG()
        
        for element_points in self.__nt:
            points = [self.__akt[i - 1] for i in element_points]
            dj_for_finite_element = []
            for index in range(27):
                dj_11 = 0 # dx/da
                dj_12 = 0 # dx/db
                dj_13 = 0 # dx/dg

                dj_21 = 0 # dy/da
                dj_22 = 0 # dy/db
                dj_23 = 0 # dy/dg

                dj_31 = 0 # dz/da
                dj_32 = 0 # dz/db
                dj_33 = 0 # dz/dg
                        
                for i in range(20):
                    point = points[i]

                    # dFi / d(a | b | g)
                    DFIDA = self.__DFIABG[index][i][0]
                    DFIDB = self.__DFIABG[index][i][1]
                    DFIDG = self.__DFIABG[index][i][2]

                    xi = point[0]
                    yi = point[1]
                    zi = point[2]

                    # print(DFIDA, DFIDB, DFIDG)

                    dj_11 += xi * DFIDA
                    dj_12 += xi * DFIDB
                    dj_13 += xi * DFIDG
                            
                    dj_21 += yi * DFIDA
                    dj_22 += yi * DFIDB
                    dj_23 += yi * DFIDG

                    dj_31 += zi * DFIDA
                    dj_32 += zi * DFIDB
                    dj_33 += zi * DFIDG

                # appending DJ for specific node
                dj_for_finite_element.append([[dj_11, dj_12, dj_13], [dj_21, dj_22, dj_23], [dj_31, dj_32, dj_33]])
                        
            dj_matrixes.append(dj_for_finite_element)
        self.__DJ_matrixes = dj_matrixes
        return dj_matrixes
    
    def DJ(self, do_health_check: bool = False):
        dj = []
        self.DJ_matrixes()
        # проходим по матрицях якобіантів для КОЖНОГО СКІНЧЕННОГО ЕЛЕМЕНТА
        for matrix_array in self.__DJ_matrixes:
            for matrix in matrix_array:
                det_matrix = matrix[0][0] * matrix[1][1] * matrix[2][2] + \
                                matrix[0][1] * matrix[1][2] * matrix[2][0] + \
                                matrix[0][2] * matrix[1][0] * matrix[2][1] - \
                                matrix[0][2] * matrix[1][1] * matrix[2][0] - \
                                matrix[0][0] * matrix[1][2] * matrix[2][1] - \
                                matrix[0][1] * matrix[1][0] * matrix[2][2]
                dj.append(det_matrix)
                
        if do_health_check:                                                                     # тут 8, бо 2*
            dj_koef = (self.__ax / self.__nx) * (self.__ay / self.__ny) * (self.__az / self.__nz) / 8
            passed = True
            for j in dj:
                if str(abs(round(j, 3))) != str(abs(round(dj_koef, 3))):
                    passed = False
            if passed:
                print("##############################")
                print("# " + CGREEN + "HEALTH CHECK FOR " + CYELLOW2 + self.DJ.__name__ + CGREEN + " PASSED" + CEND + " #")
                print("##############################")
            else:
                print("##############################")
                print("# " + CRED + "HEALTH CHECK FOR " + CYELLOW2 + self.DJ.__name__ + CRED + " FAILED" + CEND + " #")
                print("################################")
                
        self.__DJ = dj
        return dj
                        
    def DFIXYZ(self):
        dfixyz = []
        
        # calculate DJ array
        self.DJ()
        
        for finite_element_index in range(self.__nt.__len__()):
            dfixyz_for_finite_element = []
            
            # прохід по точках Гаусса
            for index in range(27):
                dfixyz_for_gaussian_point = []
                
                dj_matrix_for_gaussian_point = self.__DJ_matrixes[finite_element_index][index]
                
                # прохід по точках скінченного елементу
                for finite_element_local_point_index in range(20):
                    dfixyz_for_local_point = []
                    
                    dfia_for_local_point = self.__DFIABG[index][finite_element_local_point_index][0]
                    dfib_for_local_point = self.__DFIABG[index][finite_element_local_point_index][1]
                    dfig_for_local_point = self.__DFIABG[index][finite_element_local_point_index][2]
                    dfiabg_for_local_point = [dfia_for_local_point, dfib_for_local_point, dfig_for_local_point]
                    
                    res = np.linalg.solve(dj_matrix_for_gaussian_point, dfiabg_for_local_point)
                    
                    dfixyz_for_local_point.extend(res)
                    
                    dfixyz_for_gaussian_point.append(dfixyz_for_local_point)
                    
                dfixyz_for_finite_element.append(dfixyz_for_gaussian_point)
                
            dfixyz.append(dfixyz_for_finite_element)
        
        self.__DFIXYZ = dfixyz        
        return dfixyz
         
    # це кусок тої формули [K] * [U] = [F]
    # це один з K (матриця жоркстості), U - вектор переміщення
    # U виглядає отак (ux1, .... ux20, uy1, .... uy20, uz1, ...., uz20), де кожне оце Ю - це переміщення відповідної точки по відповідній координаті
    # загальне U буде 51*3 для дефолтного прикладу 
    def MGE(self, do_health_check: bool = False):
        self.DFIXYZ()
        
        # 60х60 для кожного скінченного елемента
        # діагональні елементи мають бути додатні і по модулі більші ніж решта елементів (30:30 практична 4)
        mge = [[] for _ in range(self.__nt.__len__())]

        heath_check_passed = True

        for finite_element_index in range(self.__nt.__len__()):            
            # всі а_х - це матриці 20х20
            # загальна матриця симетрична, нам треба тільки верхня права половина з діагоналлю
            a_11 = [[0.0 for _ in range(20)] for _ in range(20)]
            a_12 = [[0.0 for _ in range(20)] for _ in range(20)]
            a_13 = [[0.0 for _ in range(20)] for _ in range(20)]
            a_22 = [[0.0 for _ in range(20)] for _ in range(20)]
            a_23 = [[0.0 for _ in range(20)] for _ in range(20)]
            a_33 = [[0.0 for _ in range(20)] for _ in range(20)]
            
            # індекс ноди гаусса, с - нода гаусса
            c_general_index = 0
            for i in range(20):
                for j in range(20):
                    for cm in range(3):
                        for cn in range(3):
                            for ck in range(3):
                                #                             по 27 детермінантів на кожен скінченний елемент
                                # print(finite_element_index, c_general_index, 27 * finite_element_index + c_general_index)
                                dj_for_gauss_node = self.__DJ[c_general_index + finite_element_index * 27]
                                
                                dfidx = self.__DFIXYZ[finite_element_index][c_general_index][i][0]
                                dfidy = self.__DFIXYZ[finite_element_index][c_general_index][i][1]
                                dfidz = self.__DFIXYZ[finite_element_index][c_general_index][i][2]
                                dfjdx = self.__DFIXYZ[finite_element_index][c_general_index][j][0]
                                dfjdy = self.__DFIXYZ[finite_element_index][c_general_index][j][1]
                                dfjdz = self.__DFIXYZ[finite_element_index][c_general_index][j][2]
                                
                                general_koef = self.__gauss_constant[cm] * self.__gauss_constant[cn] * self.__gauss_constant[ck] * dj_for_gauss_node
                                # if dj_for_gauss_node < 0:
                                #     print(dj_for_gauss_node)
                                
                                # лаб практикум ст.13
                                a_11[i][j] += general_koef * (self.__lambda * (1 - self.__v) * (dfidx * dfjdx) + self.__mu * (dfidy * dfjdy + dfidz * dfjdz))
                                a_12[i][j] += -general_koef * (self.__lambda * self.__v * dfidx * dfjdy + self.__mu * dfidy * dfjdx)
                                a_13[i][j] += general_koef * (self.__lambda * self.__v * dfidx * dfjdz + self.__mu * dfidz * dfjdx)
                                
                                a_22[i][j] += general_koef * (self.__lambda * (1 - self.__v) * (dfidy * dfjdy) + self.__mu * (dfidx * dfjdx + dfidz * dfjdz))
                                a_23[i][j] += general_koef * (self.__lambda * self.__v * (dfidy * -dfjdz) + self.__mu * (-dfidz * dfjdy))
                                
                                a_33[i][j] += general_koef * (self.__lambda * (1 - self.__v) * (dfidz * dfjdz) + self.__mu * (dfidx * dfjdx + dfidy * dfjdy))
                                
                                # if i == j and cm == 2 and cn == 2 and ck == 2:
                                #     print(a_11[i][j], a_12[i][j], a_13[i][j], a_22[i][j], a_23[i][j], a_33[i][j])
                                
                                c_general_index += 1
                    
                    c_general_index = 0
            
            
            # print(a_11)
            # print(a_12)
            # print(a_13)
            # print(a_22)
            # print(a_23)
            # print(a_33)
            
            finite_element_mge = [[0.0 for _ in range(60)] for _ in range(60)]
            # i hate python            
            # finite_element_mge[:20, :20] = a_11
            # finite_element_mge[:20, 20:40] = a_12
            # finite_element_mge[:20, 40:] = a_13
            # finite_element_mge[20:40, 20:40] = a_22
            # finite_element_mge[20:40, 40:] = a_23
            # finite_element_mge[40:, 40:] = a_33
            # ше й не працює
            
            # розумних тут не люблять
            for i in range(20):
                for j in range(20):
                    finite_element_mge[i][j] = a_11[i][j]
                    finite_element_mge[i][j + 20] = a_12[i][j]
                    finite_element_mge[i][j + 40] = a_13[i][j]
                    finite_element_mge[i + 20][j + 20] = a_22[i][j]
                    finite_element_mge[i + 20][j + 40] = a_23[i][j]
                    finite_element_mge[i + 40][j + 40] = a_33[i][j]
                    
                    finite_element_mge[j][i] = a_11[i][j]
                    finite_element_mge[j + 20][i] = a_12[i][j]
                    finite_element_mge[j + 40][i] = a_13[i][j]
                    finite_element_mge[j + 20][i + 20] = a_22[i][j]
                    finite_element_mge[j + 40][i + 20] = a_23[i][j]
                    finite_element_mge[j + 40][i + 40] = a_33[i][j]
                    
            # print(finite_element_mge)
                
            if do_health_check:
                diagonal_sum = 0.0
                all_sum = 0.0
                for i in range(60):
                    if finite_element_mge[i][i] < 0:
                        print("##################################################################")
                        print("# " + CRED + "HEALTH CHECK FOR " + CYELLOW2 + self.MGE.__name__ + CRED + " FAILED. NOT ALL DIAGONAL ELEMENTS POSITIVE " + CEND + " #")
                        print("##################################################################")
                        heath_check_passed = False
                    
                    # print(finite_element_mge[i][i])
                        
                    diagonal_sum += finite_element_mge[i][i]
                    for j in range(60):
                        all_sum += finite_element_mge[i][j]
                all_sum -= diagonal_sum
                all_sum *= 2 # компенсовуємо шо в нас нема нижнього лівого куска
                
                if diagonal_sum <= all_sum:
                    print("####################################################")
                    print("# " + CRED + "HEALTH CHECK FOR " + CYELLOW2 + self.MGE.__name__ + CRED + " FAILED FOR FINITE ELEMENT " + str(finite_element_index + 1) + CEND + " #")
                    print("####################################################")
                    heath_check_passed = False
                if heath_check_passed:
                    print("#######################################################")
                    print("# " + CGREEN + "HEALTH CHECK FOR " + CYELLOW2 + self.MGE.__name__ + CGREEN + " PASSED FOR ALL FINITE ELEMENTS" + CEND + " #")
                    print("#######################################################")
            mge[finite_element_index] = finite_element_mge
        # print(mge[0])
        self.__MGE = mge
        return mge
    
    #-------------------------------------------------------------------------
    # слайд навантаження на систему, заняття 5 в практикумі
    # вектор навантаження 
    # має вийти вектор 24 елемента (по 8 на x y z )
    def FE(self): 
        coordinates = [
            [-1, -1],   # 1
            [1, -1],    # 2
            [1, 1],     # 3
            [-1, 1],    # 4
            [0, -1],    # 5
            [1, 0],     # 6
            [0, 1],     # 7
            [-1, 0]     # 8
        ]
        
        f = [[] for _ in range(self.__nt.__len__())]
        for finite_element_index, finite_element in enumerate(self.__nt):
            fe = [0 for _ in range(60)]
            
            # берем тільки точки, на які діє сила, решта точок будуть мати 0
            # берем кожен СЕ окремо, тому з загально зп дістаєм тільки точки цього СЕ
            # print(self.__zp[0][0])
            zp_point_for_this_finite_element = self.__zp[finite_element_index]
            # print(zp_point_for_this_finite_element)  # 4 times here
            if(zp_point_for_this_finite_element.__len__() == 0):
                fe = [0.0 for _ in range(60)]
            else:
                # print('gege', zp_point_for_this_finite_element.__len__(), zp_point_for_this_finite_element)
                for m in range(3):
                    for n in range(3):
                        eta = self.__gauss_points[m]
                        tau = self.__gauss_points[n]
                        # print(zp_point_for_this_finite_element)
                        # print('face', zp_point_for_this_finite_element[0][2])
                        finite_element_face_points = self.get_nt_point_for_face(zp_point_for_this_finite_element[0][2])
                        force = zp_point_for_this_finite_element[0][3]
                        # print(zp_point)
                        # print(zp_point_for_this_finite_element)
                        point_for_derivatives = [point_data[0] for point_data in zp_point_for_this_finite_element]
                        # print(point_for_derivatives)
                            
                        d_x_d_eta = 0
                        d_y_d_eta = 0
                        d_z_d_eta = 0
                        d_x_d_tau = 0
                        d_y_d_tau = 0
                        d_z_d_tau = 0
                            
                        # calculate d(x y z) / d (eta tau)
                        for point_index, point in enumerate(coordinates): # (36) в практикумі
                            derivative_tau = 0
                            derivative_eta = 0
                            etai = point[0]
                            taui = point[1]
                            if point_index < 4:
                                derivative_tau = self.get_dф_for_1_4_for_tau(eta, etai, tau, taui)
                                derivative_eta = self.get_dф_for_1_4_for_eta(eta, etai, tau, taui)
                            elif point_index == 4 or point_index == 6: # for 5 and 7 
                                derivative_tau = self.get_dф_for_5_and_7_for_tau(eta, etai, tau, taui)
                                derivative_eta = self.get_dф_for_5_and_7_for_eta(eta, etai, tau, taui)
                            elif point_index == 5 or point_index == 7: # for 6 and 8 
                                derivative_tau = self.get_dф_for_6_and_8_for_tau(eta, etai, tau, taui)
                                derivative_eta = self.get_dф_for_6_and_8_for_eta(eta, etai, tau, taui)
                            
                            # print(derivative_eta, derivative_tau)
                            # print(self.__akt[point_for_derivatives[point_index] - 1][0], self.__akt[point_for_derivatives[point_index] - 1][1], self.__akt[point_for_derivatives[point_index] - 1][2])
                            
                            d_x_d_eta += derivative_eta * self.__akt[point_for_derivatives[point_index] - 1][0]
                            d_y_d_eta += derivative_eta * self.__akt[point_for_derivatives[point_index] - 1][1]
                            d_z_d_eta += derivative_eta * self.__akt[point_for_derivatives[point_index] - 1][2]
                            d_x_d_tau += derivative_tau * self.__akt[point_for_derivatives[point_index] - 1][0]
                            d_y_d_tau += derivative_tau * self.__akt[point_for_derivatives[point_index] - 1][1]
                            d_z_d_tau += derivative_tau * self.__akt[point_for_derivatives[point_index] - 1][2]
                            
                        # print(d_x_d_eta)
                        # print(d_y_d_eta)
                        # print(d_z_d_eta)
                        # print(d_x_d_tau)
                        # print(d_y_d_tau)
                        # print(d_z_d_tau)    

                        for i in range(8):
                            # print(f"{finite_element_index} => {iinnddeexx} => {zp_point_index} => {i}")
                            etai = coordinates[i][0]
                            taui = coordinates[i][1]
                                    
                            ф = 0
                            if i < 4:
                                # print(eta, etai, tau, taui)
                                ф = self.get_ф_for_1_to_4(eta, etai, tau, taui)
                                # print(ф)
                            elif i == 4 or i == 6:
                                ф = self.get_ф_for_5_and_7(eta, etai, tau, taui)
                                # print('-'* 20)
                            elif i == 5 or i == 7:
                                ф = self.get_ф_for_6_and_8(eta, etai, tau, taui)
                                
                            cosnx = d_y_d_eta * d_z_d_tau - d_z_d_eta * d_y_d_tau
                            cosny = d_z_d_eta * d_x_d_tau - d_x_d_eta * d_z_d_tau
                            cosnz = d_x_d_eta * d_y_d_tau - d_y_d_eta * d_x_d_tau
                                
                            # print(f"x => {cosnx}")  
                            # print(f"y => {cosny}")  
                            # print(f"z => {cosnz}")  
                            
                            same_fe_coeff_for_x_y_z = self.__gauss_constant[m] * self.__gauss_constant[n] * force * ф 
                            # print(same_fe_coeff_for_x_y_z * cosnx)
                            fe[finite_element_face_points[i]] += same_fe_coeff_for_x_y_z * cosnx
                            fe[finite_element_face_points[i] + 20] += same_fe_coeff_for_x_y_z * cosny
                            fe[finite_element_face_points[i] + 40] += same_fe_coeff_for_x_y_z * cosnz
            # print(fe)
            f[finite_element_index] = fe
        self.__FE = f         
        return f
            
    @staticmethod
    def get_ф_for_1_to_4(eta, etai, tau, taui):
        return (1 / 4) * (1 + tau * taui) * (1 + eta * etai) * (tau * taui + eta * etai - 1)
    
    @staticmethod
    def get_ф_for_5_and_7(eta, etai, tau, taui):
        return (1 / 2) * (1 - eta * eta) * (1 + tau * taui)
    
    @staticmethod
    def get_ф_for_6_and_8(eta, etai, tau, taui):
        return (1 / 2) * (1 - tau * tau) * (1 + eta * etai)
    
    @staticmethod
    def get_dф_for_1_4_for_eta(eta, etai, tau, taui):
        return (1 / 4) * etai * (tau * taui + 1) * (2 * eta * etai + tau * taui)
    
    @staticmethod
    def get_dф_for_1_4_for_tau(eta, etai, tau, taui):
        return (1 / 4) * taui * (eta * etai + 1) * (2 * tau * taui + eta * etai)
    
    @staticmethod
    def get_dф_for_5_and_7_for_tau(eta, etai, tau, taui):
        return (1 / 2) * taui * (1 - eta * eta)
    
    @staticmethod
    def get_dф_for_5_and_7_for_eta(eta, etai, tau, taui):
        return -1 * eta * (tau * taui + 1)
    
    @staticmethod
    def get_dф_for_6_and_8_for_tau(eta, etai, tau, taui):
        return -1 * tau * (eta * etai + 1)
    
    @staticmethod
    def get_dф_for_6_and_8_for_eta(eta, etai, tau, taui):
        return (1 / 2) * etai * (1 - tau * tau)
    #-------------------------------------------------------------------------

    def MGE_and_FE_general(self):
        self.MGE()
        self.FE()
                
        verticies_count = self.__akt.__len__()
        
        mg_general = [[0.0 for _ in range(3 * verticies_count)] for _ in range(3 * verticies_count)]
        f_general = [0.0 for _ in range(3 * verticies_count)]
        
        for finite_element_index, finite_element in enumerate(self.__nt):
            finite_element_mge = self.__MGE[finite_element_index]
            finite_element_fe = self.__FE[finite_element_index]
            # print(finite_element_mge[20])
            for i in range(60):
                #                            бо 20x 20y 20z 
                #                                 \/                   
                row_index = 3 * (finite_element[i % 20] - 1) + floor(i / 20)
                f_general[row_index] += finite_element_fe[i]
                
                for j in range(60):
                    col_index = 3 * (finite_element[j % 20] - 1) + floor(j / 20)
                    # if(row_index == 1 and col_index == 0):
                    #     print(f"mge for 1 0 += {finite_element_mge[i][j]}, i = {i}, j = {j}")
                    mg_general[row_index][col_index] += finite_element_mge[i][j]
        # print(f_general)
        
        self.__MG_GENERAL = mg_general
        self.__F_GENERAL = f_general
        
        return [mg_general, f_general]

    def Apply_ZU_support(self):
        mg_general_with_support = self.__MG_GENERAL
        for point in self.__zu:
            for i in range(3):
                mg_general_with_support[3 * point - 3 + i][3 * point - 3 + i] = float('inf')
                
        self.__MG_GENERAL = mg_general_with_support
                
    def Get_Deformation_Info(self):
        self.MGE_and_FE_general()
        self.Apply_ZU_support()
        # print(self.__MG_GENERAL)                 
                
        # for row in self.__MGE[1]:
        #     print(row)
            
        # print(self.__MG_GENERAL)
        # print(self.__F_GENERAL)
        deformation_info = np.linalg.solve(self.__MG_GENERAL, self.__F_GENERAL)
        deformation_info = [round(value, 40) for value in deformation_info]
        self.__Deformation_Info = deformation_info
        
        # print(deformation_info)
        
        return deformation_info
# endregion

# region 3dObjectGenerator
class SpacialObjectGenerator:
    def __init__(self, akt, deformed_akt, nt):
        self.akt = akt
        self.deformed_akt = deformed_akt
        self.nt = nt

    def BuildImage(self):
        fig = plt.figure()
        # Розділяємо координати точок на x, y, z
        x = [point[0] for point in self.akt]
        y = [point[1] for point in self.akt]
        z = [point[2] for point in self.akt]
        
        dx = [point.x for point in self.deformed_akt]
        dy = [point.y for point in self.deformed_akt]
        dz = [point.z for point in self.deformed_akt]

        # Відображаємо точки
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z)

        # додаємо нумерацію точок
        # for i, point in enumerate(akt):
        #     ax.text(point[0], point[1], point[2], str(i+1), color='r')

        # З'єднуємо точки відповідно до індексів у nt
        for element in self.nt:
            face_points = [[element[0], element[1], element[2], element[3]],
                            [element[4], element[5], element[6], element[7]],
                            [element[1], element[2], element[6], element[5]],
                            [element[0], element[3], element[7], element[4]],
                            [element[0], element[1], element[5], element[4]],
                            [element[2], element[3], element[7], element[6]]]

            faces = [[self.akt[point - 1] for point in face] for face in face_points]
            ax.add_collection3d(Poly3DCollection(faces, color=(0, 0, 1), linewidths=1, edgecolors='k', alpha=0.1))

        ax.scatter(dx, dy, dz)
        for element in self.nt:
            bottom_face = [self.deformed_akt[element[0] - 1], self.deformed_akt[element[8] - 1], self.deformed_akt[element[1] - 1], self.deformed_akt[element[9] - 1], self.deformed_akt[element[2] - 1], self.deformed_akt[element[10] - 1], self.deformed_akt[element[3] - 1], self.deformed_akt[element[11] - 1]]
            top_face =    [self.deformed_akt[element[4] - 1], self.deformed_akt[element[16] - 1], self.deformed_akt[element[5] - 1], self.deformed_akt[element[17] - 1], self.deformed_akt[element[6] - 1], self.deformed_akt[element[18] - 1], self.deformed_akt[element[7] - 1], self.deformed_akt[element[19] - 1]]
            right_face =  [self.deformed_akt[element[1] - 1], self.deformed_akt[element[9] - 1], self.deformed_akt[element[2] - 1], self.deformed_akt[element[14] - 1], self.deformed_akt[element[6] - 1], self.deformed_akt[element[17] - 1], self.deformed_akt[element[5] - 1], self.deformed_akt[element[1] - 1]]
            left_face =   [self.deformed_akt[element[0] - 1], self.deformed_akt[element[11] - 1], self.deformed_akt[element[3] - 1], self.deformed_akt[element[15] - 1], self.deformed_akt[element[7] - 1], self.deformed_akt[element[19] - 1], self.deformed_akt[element[4] - 1], self.deformed_akt[element[12] - 1]]
            front_face =  [self.deformed_akt[element[0] - 1], self.deformed_akt[element[8] - 1], self.deformed_akt[element[1] - 1], self.deformed_akt[element[13] - 1], self.deformed_akt[element[5] - 1], self.deformed_akt[element[16] - 1], self.deformed_akt[element[4] - 1], self.deformed_akt[element[12] - 1]]
            back_face =   [self.deformed_akt[element[2] - 1], self.deformed_akt[element[10] - 1], self.deformed_akt[element[3] - 1], self.deformed_akt[element[15] - 1], self.deformed_akt[element[7] - 1], self.deformed_akt[element[18] - 1], self.deformed_akt[element[6] - 1], self.deformed_akt[element[2] - 1]]
            
            face_points = [bottom_face, top_face, right_face, left_face, front_face, back_face]
            
            # print(f"{point.x} {point.y} {point.z}" for point in face_points)
            
            deformed_faces = [[[point.x, point.y, point.z] for point in face] for face in face_points]
            ax.add_collection3d(Poly3DCollection(deformed_faces, color=(1, 0, 0), linewidths=1, edgecolors='k', alpha=0.1))


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.axis('scaled')
        plt.show()


# endregion


class DeformedPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

if __name__ == '__main__':
    ax: int = 10
    ay: int = 2
    az: int = 2

    nx: int = 5
    ny: int = 2
    nz: int = 2

    points_calculator = PointCalculator(ax, ay, az, nx, ny, nz)
    akt: list[tuple[float, float, float]] = points_calculator.AKT()
    nt: list[list[int]] = points_calculator.NT()

    # print(nt)

    dj = points_calculator.DJ()
    # print(dj)

    matr = points_calculator.DJ_matrixes();
    # print(matr)

    DFIABG = points_calculator.DFIABG()
    # print(DFIABG)

    DFIXYZ = points_calculator.DFIXYZ()
    # print(DFIXYZ)

    MGE = points_calculator.MGE()
    
    # for row in MGE:
    #     print(row)
    #     print("\n")
    #     print("\n")
    #     print("\n")

    # FE = points_calculator.FE()
    
    # GENERAL = points_calculator.MGE_and_FE_general()
    
    # print(GENERAL[0])
    
    deformation = points_calculator.Get_Deformation_Info()
    # print(deformation)
    # pprint(nt)        
    
    # print(deformation)
    
    deformed_points = []
    for i in range(int(deformation.__len__() / 3)):
        # deformed_points[i][0] += deformation[i * 3]
        # deformed_points[i][1] += deformation[i * 3 + 1]
        # deformed_points[i][1] += deformation[i * 3 + 2]
        new_point = DeformedPoint(
                akt[i][0] + deformation[i * 3],
                akt[i][1] + deformation[i * 3 + 1], 
                akt[i][2] + deformation[i * 3 + 2]
            )
        # print(new_point.x, new_point.y, new_point.z)
        deformed_points.append(new_point)
        
    # for i in range(akt.__len__()):
    #     print(str(akt[i][0] == deformed_points[i][0]) + "      " + str(akt[i][0]) + " ==> " + str(deformed_points[i][0]))
    #     print(str(akt[i][1] == deformed_points[i][1]) + "      " + str(akt[i][1]) + " ==> " + str(deformed_points[i][1]))
    #     print(str(akt[i][2] == deformed_points[i][2]) + "      " + str(akt[i][2]) + " ==> " + str(deformed_points[i][2]))
        
    SpacialObjectGenerator(akt, deformed_points, nt).BuildImage()
