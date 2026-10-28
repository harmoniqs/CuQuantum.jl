/* Minimal stand-in for cuComplex.h (see library_types.h). */
#pragma once
typedef struct { float x, y; } cuFloatComplex;
typedef struct { double x, y; } cuDoubleComplex;
typedef cuFloatComplex cuComplex;
