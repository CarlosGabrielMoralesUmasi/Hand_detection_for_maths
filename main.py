import pickle
import cv2
import mediapipe as mp 
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Configuración de Mediapipe y modelo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9)

# Etiquetas para los números y operadores
labels = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10: "+", 11: "-", 12: "*", 13: "/"}

# Cargar el modelo entrenado
with open("./rf_model.p", "rb") as f:
    model = pickle.load(f)
rf_model = model["model"]

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Función para realizar la operación matemática
def calculate(a, b, op):
    try:
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            return a / b if b != 0 else "Error: Div/0"
    except Exception as e:
        return f"Error: {e}"

# Función para verificar si el dedo índice está sobre un operador
def check_operator_selection(x, y, positions, box_size):
    for op, pos in positions.items():
        op_x, op_y = pos
        if op_x - box_size // 2 <= x <= op_x + box_size // 2 and op_y - box_size // 2 <= y <= op_y + box_size // 2:
            return op
    return None

# Ciclo principal
while True:
    # Variables para almacenar números, el operador, y temporizadores para cada cálculo
    first_number = None
    second_number = None
    operator = None
    result = None
    state = "detect_first_number"
    operation_completed = False

    # Tiempo de espera y temporizadores
    waiting_time = 3  # Tiempo de espera después de detectar el operador (en segundos)
    display_result_time = 5  # Tiempo adicional para observar el resultado (en segundos)
    display_second_number_time = 3  # Tiempo para mostrar el segundo número antes de calcular (en segundos)
    last_operator_time = None
    confirm_number_time = 1  # Tiempo que debe mantenerse un número para confirmarlo (en segundos)
    last_detected_time = 0
    last_detected_value = None
    last_second_number_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Obtener dimensiones del frame
        height, width, _ = frame.shape

        # Calcular tamaño y posición de los cuadros de operadores en función del tamaño de la ventana
        operator_box_size = int(min(width, height) * 0.1)
        operator_positions = {
            '+': (int(width * 0.3), int(height * 0.2)),
            '-': (int(width * 0.5), int(height * 0.2)),
            '*': (int(width * 0.7), int(height * 0.2)),
            '/': (int(width * 0.9), int(height * 0.2))
        }

        # Inicialización de listas y conversión de color
        normalized_landmarks = []
        x_coordinates, y_coordinates = [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con Mediapipe
        processed_image = hands.process(frame_rgb)
        hand_landmarks = processed_image.multi_hand_landmarks

        # Verificar si hay dos manos levantadas y omitir la detección
        if hand_landmarks and len(hand_landmarks) > 1:
            # Mostrar mensaje indicando que ambas manos están levantadas y omitir la detección
            cv2.putText(frame, "Please lower one hand to continue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Math Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Dibujar los operadores en pantalla, centrados en cuadros
        for op, pos in operator_positions.items():
            op_x, op_y = pos
            # Dibujar el recuadro para el operador
            cv2.rectangle(frame, (op_x - operator_box_size // 2, op_y - operator_box_size // 2),
                          (op_x + operator_box_size // 2, op_y + operator_box_size // 2), (255, 0, 0), 2)
            # Dibujar el operador dentro del recuadro
            cv2.putText(frame, op, (op_x - operator_box_size // 4, op_y + operator_box_size // 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, operator_box_size / 50, (0, 255, 0), 3)

        # Mostrar el estado actual en pantalla
        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Continuar solo si no se ha completado la operación
        if not operation_completed and hand_landmarks:
            for hand_landmark in hand_landmarks:
                # Dibujar los puntos y conexiones de la mano
                mp_drawing.draw_landmarks(
                    frame, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extraer coordenadas normalizadas
                for lm in hand_landmark.landmark:
                    x_coordinates.append(lm.x)
                    y_coordinates.append(lm.y)

                min_x, min_y = min(x_coordinates), min(y_coordinates)
                for lm in hand_landmark.landmark:
                    normalized_x = lm.x - min_x
                    normalized_y = lm.y - min_y
                    normalized_landmarks.extend((normalized_x, normalized_y))

                # Limitar a las primeras 42 características antes de la predicción
                sample = np.asarray(normalized_landmarks[:42]).reshape(1, -1)

                # Obtener posición del dedo índice
                index_finger_tip_x = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                index_finger_tip_y = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

                # Detección y confirmación del número (3 segundos de persistencia)
                pred = rf_model.predict(sample)
                predicted_character = labels[int(pred[0])]
                
                if state == "detect_first_number":
                    if predicted_character.isdigit():
                        # Confirmar el número si se mantiene por 3 segundos
                        if predicted_character == last_detected_value:
                            if time.time() - last_detected_time >= confirm_number_time:
                                first_number = int(predicted_character)
                                state = "select_operator"
                        else:
                            last_detected_value = predicted_character
                            last_detected_time = time.time()

                elif state == "select_operator":
                    # Verificar si el dedo índice está sobre un operador
                    selected_operator = check_operator_selection(index_finger_tip_x, index_finger_tip_y, operator_positions, operator_box_size)
                    if selected_operator:
                        operator = selected_operator
                        last_operator_time = time.time()
                        state = "waiting"  # Cambiar al estado de espera antes de detectar el segundo número

                elif state == "waiting":
                    # Esperar antes de detectar el segundo número
                    if time.time() - last_operator_time >= waiting_time:
                        state = "detect_second_number"

                elif state == "detect_second_number":
                    # Confirmar el segundo número si se mantiene por 3 segundos
                    if predicted_character.isdigit():
                        if predicted_character == last_detected_value:
                            if time.time() - last_detected_time >= confirm_number_time:
                                second_number = int(predicted_character)
                                last_second_number_time = time.time()  # Guardar el tiempo de detección del segundo número
                                state = "show_second_number"
                        else:
                            last_detected_value = predicted_character
                            last_detected_time = time.time()

                elif state == "show_second_number":
                    # Mostrar el segundo número por 2 segundos antes de calcular el resultado
                    if time.time() - last_second_number_time >= display_second_number_time:
                        result = calculate(first_number, second_number, operator)
                        operation_completed = True  # Marcar operación como completada
                        last_operator_time = time.time()

            # Mostrar los valores detectados en la parte inferior izquierda
            cv2.putText(frame, f"First Number: {first_number if first_number is not None else ''}", (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Operator: {operator if operator is not None else ''}", (10, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Second Number: {second_number if second_number is not None else ''}", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el resultado extendido en la parte inferior izquierda
        if operation_completed:
            cv2.putText(frame, f"Result: {result}", (10, height - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if time.time() - last_operator_time >= display_result_time:
                break  # Salir del ciclo interno para reiniciar el cálculo

        # Mostrar el frame con la detección
        cv2.imshow("Math Detection", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
