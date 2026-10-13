#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conteo de puntos verdes fluorescentes dentro de los 16 cuadros (4 x 4) de una
cámara de conteo. No cuenta los puntos de las zonas de líneas apretadas de los
extremos ni los que caen sobre las líneas.

Uso:
    python contar_puntos.py CARPETA_IMAGENES
    python contar_puntos.py CARPETA_IMAGENES -o resultados.xlsx --umbral 30

Salida:
    - Excel (por defecto CARPETA_IMAGENES/conteos.xlsx) con dos hojas:
        "Resumen"    -> imagen, recuento total en los 16 cuadros, promedio por cuadro
        "Por_cuadro" -> recuento de cada cuadro (filas A-D de arriba abajo, columnas 1-4
                        de izquierda a derecha)
    - Carpeta "control/" junto al Excel con cada imagen anotada (cuadrícula detectada,
      recuento por cuadro y cada punto contado marcado en rojo) para revisión visual.

Requisitos:
    pip install opencv-python numpy scipy scikit-image pandas openpyxl
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------
# PARÁMETROS (ajustables)
# ----------------------------------------------------------------------------
EXTENSIONES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

ANCHO_REF = 1600          # las imágenes se reescalan a este ancho para que los
                          # parámetros en píxeles sean consistentes entre fotos
# --- Detección de la cuadrícula
UMBRAL_LINEA = 25         # brillo sobre el fondo para que un píxel cuente como "línea"
LARGO_MIN_LINEA = 101     # largo mínimo (px) para considerar un segmento como línea
ESPACIADO_MIN = 70        # lado mínimo / máximo esperado (px) de cada cuadro grande
ESPACIADO_MAX = 260
ANGULOS_PRUEBA = np.arange(-6, 6.5, 1.0)   # rotaciones (°) que se prueban si la foto está girada
# --- Detección de puntos
TOPHAT_KERNEL = 9         # algo mayor que el diámetro de un punto (px)
UMBRAL_PUNTO = 25         # intensidad mínima (sobre el fondo local) para contar un punto
DIST_MIN_PUNTOS = 2       # separación mínima (px) entre dos puntos distintos
MARGEN_LINEA = 3          # px extra que se excluyen alrededor de cada línea


# ----------------------------------------------------------------------------
# Utilidades de imagen
# ----------------------------------------------------------------------------
def leer_imagen(ruta: Path) -> np.ndarray:
    """Lee la imagen (funciona con rutas con tildes en Windows) y la reescala."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("no se pudo leer la imagen")
    escala = ANCHO_REF / img.shape[1]
    if abs(escala - 1) > 1e-3:
        img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return img


def rotar(img: np.ndarray, angulo: float) -> np.ndarray:
    if abs(angulo) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def mascaras_lineas(img: np.ndarray):
    """Máscaras de líneas horizontales y verticales largas (canal verde, sin fondo)."""
    g = img[:, :, 1].astype(np.float32)
    d = g - cv2.GaussianBlur(g, (0, 0), 25)
    binaria = (d > UMBRAL_LINEA).astype(np.uint8) * 255
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (LARGO_MIN_LINEA, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LARGO_MIN_LINEA))
    horiz = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kh)
    vert = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kv)
    return horiz, vert


# ----------------------------------------------------------------------------
# Detección de la cuadrícula 4 x 4
# ----------------------------------------------------------------------------
def buscar_rejilla_4x4(perfil: np.ndarray):
    """
    En un perfil de proyección busca 5 líneas equiespaciadas (que delimitan 4 cuadros)
    cuyos intervalos estén vacíos. Las zonas de líneas apretadas de los extremos se
    descartan porque tienen líneas dentro de cada intervalo.
    Devuelve (lista_de_5_posiciones, puntaje) o (None, -inf).
    """
    perfil = cv2.GaussianBlur(perfil.reshape(1, -1).astype(np.float32), (0, 0), 1.5).ravel()
    if perfil.max() <= 0:
        return None, -np.inf
    picos, _ = find_peaks(perfil, height=perfil.max() * 0.15, distance=6)
    n = len(perfil)
    mejor, mejor_lineas = -np.inf, None
    for i in range(len(picos)):
        for j in range(i + 1, len(picos)):
            s = picos[j] - picos[i]
            if not (ESPACIADO_MIN <= s <= ESPACIADO_MAX):
                continue
            tol = max(4, 0.06 * s)
            for k0 in range(4):          # posición de picos[i] dentro de las 5 líneas
                lineas = []
                for k in range(5):
                    esperado = picos[i] + (k - k0) * s
                    cerca = picos[np.abs(picos - esperado) <= tol]
                    if esperado < 0 or esperado >= n or len(cerca) == 0:
                        break
                    lineas.append(int(cerca[np.argmax(perfil[cerca])]))
                if len(lineas) < 5 or len(set(lineas)) < 5:
                    continue
                lineas = sorted(lineas)
                gaps = np.diff(lineas)
                if gaps.min() < 0.85 * gaps.max():
                    continue
                fuerza = perfil[lineas].min()
                interior = 0.0
                for a, b in zip(lineas[:-1], lineas[1:]):
                    m = int(0.12 * (b - a))
                    interior = max(interior, perfil[a + m:b - m].max())
                if interior > 0.5 * fuerza:
                    continue                 # hay líneas dentro: zona de líneas apretadas
                score = fuerza - 2.0 * interior
                if score > mejor:
                    mejor, mejor_lineas = score, lineas
    return mejor_lineas, mejor


def ajustar_curva(mascara: np.ndarray, pos: int, vertical: bool, banda: int, rango):
    """
    Ajusta un polinomio de 2.º grado a los píxeles de una línea cercanos a 'pos'
    (tolera la leve curvatura que introduce la óptica del microscopio/celular).
    Vertical:   x = f(y)      Horizontal: y = f(x)
    """
    a0, a1 = rango
    lo = max(0, pos - banda)
    if vertical:
        ys, xs = np.nonzero(mascara[a0:a1, lo:pos + banda + 1])
        t, u = ys + a0, xs + lo
    else:
        ys, xs = np.nonzero(mascara[lo:pos + banda + 1, a0:a1])
        t, u = xs + a0, ys + lo
    if len(t) < 30:
        return np.poly1d([float(pos)])
    grado = 2 if (t.max() - t.min()) > 0.6 * (a1 - a0) else 1
    coef = np.polyfit(t, u, grado)
    # una pasada de rechazo de atípicos
    res = np.abs(np.polyval(coef, t) - u)
    ok = res < max(2.0, 2.5 * np.median(res))
    if ok.sum() > 30:
        coef = np.polyfit(t[ok], u[ok], grado)
    return np.poly1d(coef)


def detectar_cuadricula(img: np.ndarray):
    """Prueba varias rotaciones y se queda con la que da la cuadrícula 4x4 más clara."""
    mejor = None
    for ang in sorted(ANGULOS_PRUEBA, key=abs):
        im_r = rotar(img, ang)
        horiz, vert = mascaras_lineas(im_r)
        xs, sx = buscar_rejilla_4x4(vert.sum(0) / 255.0)
        ys, sy = buscar_rejilla_4x4(horiz.sum(1) / 255.0)
        if xs is None or ys is None:
            continue
        score = min(sx, sy)
        if mejor is None or score > mejor[0]:
            mejor = (score, ang, im_r, horiz, vert, xs, ys)
    if mejor is None:
        raise RuntimeError("no se detectó la cuadrícula 4x4")
    _, ang, im_r, horiz, vert, xs, ys = mejor

    banda = int(0.08 * min(np.diff(xs).mean(), np.diff(ys).mean()))
    margen = int(0.5 * np.diff(ys).mean())
    lv = [ajustar_curva(vert, x, True, banda, (max(0, ys[0] - margen), ys[-1] + margen)) for x in xs]
    margen = int(0.5 * np.diff(xs).mean())
    lh = [ajustar_curva(horiz, y, False, banda, (max(0, xs[0] - margen), xs[-1] + margen)) for y in ys]
    return ang, im_r, horiz, vert, lv, lh


# ----------------------------------------------------------------------------
# Procesamiento de una imagen
# ----------------------------------------------------------------------------
def procesar(ruta: Path, carpeta_control: Path | None):
    img = leer_imagen(ruta)
    ang, img, horiz, vert, lv, lh = detectar_cuadricula(img)

    # Máscara de exclusión: las propias líneas (engrosadas)
    lineas = cv2.bitwise_or(horiz, vert)
    t = 2 * MARGEN_LINEA + 1
    lineas = cv2.dilate(lineas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))

    # Detección de puntos: top-hat (resalta objetos pequeños y brillantes) + máximos locales
    g = img[:, :, 1].astype(np.float32)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL)))
    th = cv2.GaussianBlur(th, (0, 0), 0.8)
    th[lineas > 0] = 0
    puntos = peak_local_max(th, min_distance=DIST_MIN_PUNTOS,
                            threshold_abs=UMBRAL_PUNTO, exclude_border=False)

    # Asignar cada punto a su cuadro según su posición respecto a las líneas
    conteos = np.zeros((4, 4), dtype=int)
    contados = []
    for y, x in puntos:
        bx = [f(y) for f in lv]          # posición x de cada línea vertical a esa altura
        by = [f(x) for f in lh]          # posición y de cada línea horizontal en esa columna
        if not (bx[0] < x < bx[4] and by[0] < y < by[4]):
            continue
        co = int(np.searchsorted(bx, x)) - 1
        fi = int(np.searchsorted(by, y)) - 1
        conteos[fi, co] += 1
        contados.append((x, y))

    if carpeta_control is not None:
        guardar_control(img, lv, lh, conteos, contados,
                        carpeta_control / f"{ruta.stem}_control.jpg")
    return conteos, ang


def guardar_control(img, lv, lh, conteos, contados, destino: Path):
    ctrl = img.copy()
    # dibujar las curvas de la cuadrícula
    y0 = min(f(x) for f in lh[:1] for x in (lv[0](0), lv[4](0)))
    y1 = max(f(x) for f in lh[4:] for x in (lv[0](0), lv[4](0)))
    tt = np.linspace(y0, y1, 60)
    for f in lv:
        pts = np.stack([f(tt), tt], 1).astype(np.int32)
        cv2.polylines(ctrl, [pts], False, (255, 0, 255), 2)
    x0 = min(f(y) for f in lv[:1] for y in (y0, y1))
    x1 = max(f(y) for f in lv[4:] for y in (y0, y1))
    tt = np.linspace(x0, x1, 60)
    for f in lh:
        pts = np.stack([tt, f(tt)], 1).astype(np.int32)
        cv2.polylines(ctrl, [pts], False, (255, 0, 255), 2)
    for x, y in contados:
        cv2.circle(ctrl, (int(x), int(y)), 3, (0, 0, 255), 1)
    # recuento en cada cuadro
    for fi in range(4):
        for co in range(4):
            cy = (lh[fi]((lv[co](0) + lv[co + 1](0)) / 2) + lh[fi + 1]((lv[co](0) + lv[co + 1](0)) / 2)) / 2
            cx = (lv[co](cy) + lv[co + 1](cy)) / 2
            txt = str(conteos[fi, co])
            org = (int(cx) - 8 * len(txt), int(cy) + 9)
            cv2.putText(ctrl, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(ctrl, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ctrl, f"Total 16 cuadros: {conteos.sum()}", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", ctrl, [cv2.IMWRITE_JPEG_QUALITY, 92])
    buf.tofile(str(destino))


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    global UMBRAL_PUNTO
    ap = argparse.ArgumentParser(description="Conteo de puntos fluorescentes en la cuadrícula 4x4.")
    ap.add_argument("carpeta", help="carpeta con las imágenes")
    ap.add_argument("-o", "--salida", default=None,
                    help="archivo Excel de salida (por defecto: <carpeta>/conteos.xlsx)")
    ap.add_argument("--umbral", type=float, default=UMBRAL_PUNTO,
                    help=f"intensidad mínima de un punto (por defecto {UMBRAL_PUNTO}); "
                         "súbalo si cuenta ruido, bájelo si omite puntos tenues")
    ap.add_argument("--sin-control", action="store_true",
                    help="no guardar las imágenes de control anotadas")
    args = ap.parse_args()
    UMBRAL_PUNTO = args.umbral

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        sys.exit(f"No existe la carpeta: {carpeta}")
    imagenes = sorted(p for p in carpeta.iterdir()
                      if p.suffix.lower() in EXTENSIONES and not p.stem.endswith("_control"))
    if not imagenes:
        sys.exit("No se encontraron imágenes en la carpeta.")

    salida = Path(args.salida) if args.salida else carpeta / "conteos.xlsx"
    salida.parent.mkdir(parents=True, exist_ok=True)
    carpeta_control = None
    if not args.sin_control:
        carpeta_control = salida.parent / "control"
        carpeta_control.mkdir(parents=True, exist_ok=True)

    resumen, detalle = [], []
    for ruta in imagenes:
        try:
            conteos, ang = procesar(ruta, carpeta_control)
            total = int(conteos.sum())
            resumen.append({"Imagen": ruta.name,
                            "Recuento total (16 cuadros)": total,
                            "Promedio por cuadro": round(total / 16, 2),
                            "Desv. est. entre cuadros": round(float(conteos.std(ddof=1)), 2),
                            "Rotación corregida (°)": float(ang),
                            "Observación": ""})
            fila = {"Imagen": ruta.name}
            for fi, letra in enumerate("ABCD"):
                for co in range(4):
                    fila[f"{letra}{co + 1}"] = int(conteos[fi, co])
            detalle.append(fila)
            print(f"{ruta.name}: {total}")
        except Exception as e:
            resumen.append({"Imagen": ruta.name, "Recuento total (16 cuadros)": None,
                            "Promedio por cuadro": None, "Desv. est. entre cuadros": None,
                            "Rotación corregida (°)": None, "Observación": f"ERROR: {e}"})
            print(f"{ruta.name}: ERROR - {e}")

    with pd.ExcelWriter(salida, engine="openpyxl") as xw:
        pd.DataFrame(resumen).to_excel(xw, sheet_name="Resumen", index=False)
        pd.DataFrame(detalle).to_excel(xw, sheet_name="Por_cuadro", index=False)
        for hoja in xw.sheets.values():
            for col in hoja.columns:
                ancho = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                hoja.column_dimensions[col[0].column_letter].width = min(max(8, ancho + 2), 50)
            hoja.freeze_panes = "B2"

    print(f"\nResultados guardados en: {salida}")
    if carpeta_control:
        print(f"Imágenes de control en:  {carpeta_control}")


if __name__ == "__main__":
    main()
