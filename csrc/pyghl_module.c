#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include "ghl.h"
#include "ghl_con2prim.h"

static PyObject *GRHayLError = NULL;

typedef struct {
  PyObject_HEAD
  ghl_parameters params;
} PyGHLParams;

typedef struct {
  PyObject_HEAD
  ghl_eos_parameters eos;
  int initialized;
} PyGHLTabulatedEOS;

typedef struct {
  PyObject_HEAD
  ghl_primitive_quantities prims;
} PyGHLPrimitive;

typedef struct {
  PyObject_HEAD
  ghl_conservative_quantities cons;
} PyGHLConservative;

typedef struct {
  PyObject_HEAD
  ghl_metric_quantities metric;
} PyGHLMetric;

typedef struct {
  PyObject_HEAD
  ghl_ADM_aux_quantities aux;
} PyGHLADMAux;

typedef struct {
  PyObject_HEAD
  ghl_con2prim_diagnostics diagnostics;
} PyGHLDiagnostics;

static PyObject *diagnostics_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);

static const char *ghl_error_name(const ghl_error_codes_t code) {
  switch(code) {
    case ghl_success:
      return "ghl_success";
    case ghl_error_unknown_eos_type:
      return "ghl_error_unknown_eos_type";
    case ghl_error_invalid_c2p_key:
      return "ghl_error_invalid_c2p_key";
    case ghl_error_neg_rho:
      return "ghl_error_neg_rho";
    case ghl_error_neg_pressure:
      return "ghl_error_neg_pressure";
    case ghl_error_neg_vsq:
      return "ghl_error_neg_vsq";
    case ghl_error_c2p_max_iter:
      return "ghl_error_c2p_max_iter";
    case ghl_error_c2p_singular:
      return "ghl_error_c2p_singular";
    case ghl_error_root_not_bracketed:
      return "ghl_error_root_not_bracketed";
    case ghl_error_table_max_rho:
      return "ghl_error_table_max_rho";
    case ghl_error_table_min_rho:
      return "ghl_error_table_min_rho";
    case ghl_error_table_max_ye:
      return "ghl_error_table_max_ye";
    case ghl_error_table_min_ye:
      return "ghl_error_table_min_ye";
    case ghl_error_table_max_T:
      return "ghl_error_table_max_T";
    case ghl_error_table_min_T:
      return "ghl_error_table_min_T";
    case ghl_error_exceed_table_vars:
      return "ghl_error_exceed_table_vars";
    case ghl_error_table_neg_energy:
      return "ghl_error_table_neg_energy";
    case ghl_error_table_bisection:
      return "ghl_error_table_bisection";
    case ghl_error_u0_singular:
      return "ghl_error_u0_singular";
    case ghl_error_invalid_utsq:
      return "ghl_error_invalid_utsq";
    case ghl_error_invalid_Z:
      return "ghl_error_invalid_Z";
    case ghl_error_newman_invalid_discriminant:
      return "ghl_error_newman_invalid_discriminant";
    default:
      return "ghl_error_unknown";
  }
}

static int raise_ghl_error(const char *context, const ghl_error_codes_t code) {
  PyErr_Format(
        GRHayLError,
        "%s failed with %s (%d)",
        context,
        ghl_error_name(code),
        (int)code);
  return -1;
}

static int parse_double(PyObject *value, double *out, const char *name) {
  if(value == NULL) {
    PyErr_Format(PyExc_TypeError, "%s cannot be NULL.", name);
    return -1;
  }
  *out = PyFloat_AsDouble(value);
  if(PyErr_Occurred()) {
    PyErr_Format(PyExc_TypeError, "%s must be a real number.", name);
    return -1;
  }
  return 0;
}

static int parse_bool(PyObject *value, bool *out, const char *name) {
  const int truthy = PyObject_IsTrue(value);
  if(truthy < 0) {
    PyErr_Format(PyExc_TypeError, "%s must be truthy or falsy.", name);
    return -1;
  }
  *out = truthy != 0;
  return 0;
}

static int parse_vector3(PyObject *value, double out[3], const char *name) {
  PyObject *seq = PySequence_Fast(value, "expected a 3-element sequence");
  if(seq == NULL) {
    PyErr_Format(PyExc_TypeError, "%s must be a 3-element sequence.", name);
    return -1;
  }

  if(PySequence_Fast_GET_SIZE(seq) != 3) {
    Py_DECREF(seq);
    PyErr_Format(PyExc_ValueError, "%s must contain exactly 3 values.", name);
    return -1;
  }

  for(Py_ssize_t i = 0; i < 3; ++i) {
    if(parse_double(PySequence_Fast_GET_ITEM(seq, i), &out[i], name) < 0) {
      Py_DECREF(seq);
      return -1;
    }
  }

  Py_DECREF(seq);
  return 0;
}

static PyObject *build_vector3(const double values[3]) {
  return Py_BuildValue("(ddd)", values[0], values[1], values[2]);
}

static PyObject *build_matrix3x3(const double values[3][3]) {
  return Py_BuildValue(
        "((ddd)(ddd)(ddd))",
        values[0][0],
        values[0][1],
        values[0][2],
        values[1][0],
        values[1][1],
        values[1][2],
        values[2][0],
        values[2][1],
        values[2][2]);
}

#define DEFINE_DOUBLE_GETTER(name, type_name, field)            \
  static PyObject *name(PyObject *self, void *closure) {        \
    (void)closure;                                              \
    return PyFloat_FromDouble(((type_name *)self)->field);      \
  }

#define DEFINE_DOUBLE_SETTER(name, type_name, field, label)             \
  static int name(PyObject *self, PyObject *value, void *closure) {     \
    double parsed = 0.0;                                                \
    (void)closure;                                                      \
    if(value == NULL) {                                                 \
      PyErr_Format(PyExc_TypeError, "Cannot delete %s.", label);        \
      return -1;                                                        \
    }                                                                   \
    if(parse_double(value, &parsed, label) < 0) {                       \
      return -1;                                                        \
    }                                                                   \
    ((type_name *)self)->field = parsed;                                \
    return 0;                                                           \
  }

#define DEFINE_BOOL_GETTER(name, type_name, field)              \
  static PyObject *name(PyObject *self, void *closure) {        \
    (void)closure;                                              \
    return PyBool_FromLong(((type_name *)self)->field);         \
  }

#define DEFINE_BOOL_SETTER(name, type_name, field, label)               \
  static int name(PyObject *self, PyObject *value, void *closure) {     \
    bool parsed = false;                                                \
    (void)closure;                                                      \
    if(value == NULL) {                                                 \
      PyErr_Format(PyExc_TypeError, "Cannot delete %s.", label);        \
      return -1;                                                        \
    }                                                                   \
    if(parse_bool(value, &parsed, label) < 0) {                         \
      return -1;                                                        \
    }                                                                   \
    ((type_name *)self)->field = parsed;                                \
    return 0;                                                           \
  }

static int eos_ensure_initialized(PyGHLTabulatedEOS *self) {
  if(self->initialized) {
    return 0;
  }
  PyErr_SetString(PyExc_RuntimeError, "TabulatedEOS is not initialized.");
  return -1;
}

static void eos_dealloc(PyObject *self) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  if(eos->initialized && ghl_tabulated_free_memory != NULL) {
    ghl_tabulated_free_memory(&eos->eos);
    eos->initialized = 0;
  }
  Py_TYPE(self)->tp_free(self);
}

static PyObject *params_get_main_routine(PyObject *self, void *closure) {
  (void)closure;
  return PyLong_FromLong(((PyGHLParams *)self)->params.main_routine);
}

static PyObject *params_get_backup_routines(PyObject *self, void *closure) {
  (void)closure;
  const PyGHLParams *params = (const PyGHLParams *)self;
  return Py_BuildValue(
        "(iii)",
        params->params.backup_routine[0],
        params->params.backup_routine[1],
        params->params.backup_routine[2]);
}

DEFINE_BOOL_GETTER(params_get_evolve_entropy, PyGHLParams, params.evolve_entropy)
DEFINE_BOOL_GETTER(params_get_evolve_temp, PyGHLParams, params.evolve_temp)
DEFINE_BOOL_GETTER(params_get_calc_prim_guess, PyGHLParams, params.calc_prim_guess)
DEFINE_DOUBLE_GETTER(params_get_psi6threshold, PyGHLParams, params.psi6threshold)
DEFINE_DOUBLE_GETTER(params_get_max_lorentz_factor, PyGHLParams, params.max_Lorentz_factor)
DEFINE_DOUBLE_GETTER(
      params_get_lorenz_damping_factor,
      PyGHLParams,
      params.Lorenz_damping_factor)

static PyObject *params_repr(PyObject *self) {
  const PyGHLParams *params = (const PyGHLParams *)self;
  char buffer[256];
  snprintf(
        buffer,
        sizeof(buffer),
        "Params(main_routine=%d, backup_routines=(%d, %d, %d), evolve_entropy=%s, "
        "evolve_temp=%s, calc_prim_guess=%s, psi6threshold=%g, max_lorentz_factor=%g, "
        "lorenz_damping_factor=%g)",
        params->params.main_routine,
        params->params.backup_routine[0],
        params->params.backup_routine[1],
        params->params.backup_routine[2],
        params->params.evolve_entropy ? "True" : "False",
        params->params.evolve_temp ? "True" : "False",
        params->params.calc_prim_guess ? "True" : "False",
        params->params.psi6threshold,
        params->params.max_Lorentz_factor,
        params->params.Lorenz_damping_factor);
  return PyUnicode_FromString(buffer);
}

static PyGetSetDef params_getset[] = {
  {"main_routine", params_get_main_routine, NULL, "Main Con2Prim method.", NULL},
  {"backup_routines",
   params_get_backup_routines,
   NULL,
   "Tuple with backup Con2Prim methods.",
   NULL},
  {"evolve_entropy", params_get_evolve_entropy, NULL, "Whether entropy is evolved.", NULL},
  {"evolve_temp", params_get_evolve_temp, NULL, "Whether temperature is evolved.", NULL},
  {"calc_prim_guess",
   params_get_calc_prim_guess,
   NULL,
   "Whether primitive guesses are computed.",
   NULL},
  {"psi6threshold", params_get_psi6threshold, NULL, "psi^6 threshold.", NULL},
  {"max_lorentz_factor",
   params_get_max_lorentz_factor,
   NULL,
   "Maximum Lorentz factor.",
   NULL},
  {"lorenz_damping_factor",
   params_get_lorenz_damping_factor,
   NULL,
   "Lorenz damping factor.",
   NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyGHLParamsType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.Params",
  .tp_basicsize = sizeof(PyGHLParams),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "GRHayL parameter bundle.",
  .tp_repr = params_repr,
  .tp_getset = params_getset,
};

DEFINE_DOUBLE_GETTER(eos_get_rho_min, PyGHLTabulatedEOS, eos.rho_min)
DEFINE_DOUBLE_GETTER(eos_get_rho_max, PyGHLTabulatedEOS, eos.rho_max)
DEFINE_DOUBLE_GETTER(eos_get_ye_min, PyGHLTabulatedEOS, eos.Y_e_min)
DEFINE_DOUBLE_GETTER(eos_get_ye_max, PyGHLTabulatedEOS, eos.Y_e_max)
DEFINE_DOUBLE_GETTER(eos_get_t_min, PyGHLTabulatedEOS, eos.T_min)
DEFINE_DOUBLE_GETTER(eos_get_t_max, PyGHLTabulatedEOS, eos.T_max)
DEFINE_DOUBLE_GETTER(eos_get_table_t_min, PyGHLTabulatedEOS, eos.table_T_min)
DEFINE_DOUBLE_GETTER(eos_get_table_t_max, PyGHLTabulatedEOS, eos.table_T_max)
DEFINE_DOUBLE_GETTER(
      eos_get_root_finding_precision,
      PyGHLTabulatedEOS,
      eos.root_finding_precision)
DEFINE_DOUBLE_SETTER(
      eos_set_root_finding_precision,
      PyGHLTabulatedEOS,
      eos.root_finding_precision,
      "root_finding_precision")
DEFINE_BOOL_GETTER(
      eos_get_enable_neural_net_c2p,
      PyGHLTabulatedEOS,
      eos.enable_neural_net_c2p)
DEFINE_BOOL_SETTER(
      eos_set_enable_neural_net_c2p,
      PyGHLTabulatedEOS,
      eos.enable_neural_net_c2p,
      "enable_neural_net_c2p")

static PyObject *eos_repr(PyObject *self) {
  const PyGHLTabulatedEOS *eos = (const PyGHLTabulatedEOS *)self;
  char buffer[160];
  snprintf(
        buffer,
        sizeof(buffer),
        "TabulatedEOS(rho=[%g, %g], Ye=[%g, %g], T=[%g, %g])",
        eos->eos.rho_min,
        eos->eos.rho_max,
        eos->eos.Y_e_min,
        eos->eos.Y_e_max,
        eos->eos.T_min,
        eos->eos.T_max);
  return PyUnicode_FromString(buffer);
}

static PyObject *eos_tabulated_enforce_bounds_rho_Ye_T(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double T;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_enforce_bounds_rho_Ye_T", &rho, &Y_e, &T)) {
    return NULL;
  }
  if(ghl_tabulated_enforce_bounds_rho_Ye_T == NULL) {
    PyErr_SetString(
          PyExc_RuntimeError,
          "Tabulated EOS function pointers are not initialized in GRHayL.");
    return NULL;
  }

  ghl_tabulated_enforce_bounds_rho_Ye_T(&eos->eos, &rho, &Y_e, &T);
  return Py_BuildValue("(ddd)", rho, Y_e, T);
}

static PyObject *eos_tabulated_enforce_bounds_rho_Ye_eps(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double eps;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(
            args,
            "ddd:tabulated_enforce_bounds_rho_Ye_eps",
            &rho,
            &Y_e,
            &eps)) {
    return NULL;
  }
  if(ghl_tabulated_enforce_bounds_rho_Ye_eps == NULL) {
    PyErr_SetString(
          PyExc_RuntimeError,
          "Tabulated EOS function pointers are not initialized in GRHayL.");
    return NULL;
  }

  ghl_tabulated_enforce_bounds_rho_Ye_eps(&eos->eos, &rho, &Y_e, &eps);
  return Py_BuildValue("(ddd)", rho, Y_e, eps);
}

static PyObject *eos_tabulated_compute_P_from_T(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double T;
  double P;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_compute_P_from_T", &rho, &Y_e, &T)) {
    return NULL;
  }

  const ghl_error_codes_t error = ghl_tabulated_compute_P_from_T(&eos->eos, rho, Y_e, T, &P);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_P_from_T", error);
    return NULL;
  }

  return PyFloat_FromDouble(P);
}

static PyObject *eos_tabulated_compute_eps_from_T(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double T;
  double eps;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_compute_eps_from_T", &rho, &Y_e, &T)) {
    return NULL;
  }

  const ghl_error_codes_t error
        = ghl_tabulated_compute_eps_from_T(&eos->eos, rho, Y_e, T, &eps);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_eps_from_T", error);
    return NULL;
  }

  return PyFloat_FromDouble(eps);
}

static PyObject *eos_tabulated_compute_cs2_from_T(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double T;
  double cs2;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_compute_cs2_from_T", &rho, &Y_e, &T)) {
    return NULL;
  }

  const ghl_error_codes_t error
        = ghl_tabulated_compute_cs2_from_T(&eos->eos, rho, Y_e, T, &cs2);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_cs2_from_T", error);
    return NULL;
  }

  return PyFloat_FromDouble(cs2);
}

static PyObject *eos_tabulated_compute_P_eps_from_T(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double T;
  double P;
  double eps;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_compute_P_eps_from_T", &rho, &Y_e, &T)) {
    return NULL;
  }

  const ghl_error_codes_t error
        = ghl_tabulated_compute_P_eps_from_T(&eos->eos, rho, Y_e, T, &P, &eps);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_P_eps_from_T", error);
    return NULL;
  }

  return Py_BuildValue("(dd)", P, eps);
}

static PyObject *eos_tabulated_compute_P_eps_S_from_T(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double T;
  double P;
  double eps;
  double entropy;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(
            args,
            "ddd:tabulated_compute_P_eps_S_from_T",
            &rho,
            &Y_e,
            &T)) {
    return NULL;
  }

  const ghl_error_codes_t error = ghl_tabulated_compute_P_eps_S_from_T(
        &eos->eos, rho, Y_e, T, &P, &eps, &entropy);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_P_eps_S_from_T", error);
    return NULL;
  }

  return Py_BuildValue("(ddd)", P, eps, entropy);
}

static PyObject *eos_tabulated_compute_T_from_eps(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double eps;
  double T;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_compute_T_from_eps", &rho, &Y_e, &eps)) {
    return NULL;
  }

  const ghl_error_codes_t error
        = ghl_tabulated_compute_T_from_eps(&eos->eos, rho, Y_e, eps, &T);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_T_from_eps", error);
    return NULL;
  }

  return PyFloat_FromDouble(T);
}

static PyObject *eos_tabulated_compute_P_T_from_eps(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  double rho;
  double Y_e;
  double eps;
  double P;
  // GRHayL uses the input temperature as an initial guess during inversion.
  // Match the C nn-testing path, which seeds this with the table maximum.
  double T = eos->eos.table_T_max;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "ddd:tabulated_compute_P_T_from_eps", &rho, &Y_e, &eps)) {
    return NULL;
  }

  const ghl_error_codes_t error
        = ghl_tabulated_compute_P_T_from_eps(&eos->eos, rho, Y_e, eps, &P, &T);
  if(error != ghl_success) {
    raise_ghl_error("tabulated_compute_P_T_from_eps", error);
    return NULL;
  }

  return Py_BuildValue("(dd)", P, T);
}

static PyObject *eos_close(PyObject *self, PyObject *Py_UNUSED(args)) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  if(eos->initialized && ghl_tabulated_free_memory != NULL) {
    ghl_tabulated_free_memory(&eos->eos);
    eos->initialized = 0;
  }
  Py_RETURN_NONE;
}

static PyObject *eos_load_nn_c2p_hdf5(PyObject *self, PyObject *args) {
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)self;
  const char *model_path = NULL;

  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "s:load_nn_c2p_hdf5", &model_path)) {
    return NULL;
  }
  if(access(model_path, R_OK) != 0) {
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError, model_path);
  }

  ghl_c2p_nn_load_hdf5(model_path, &eos->eos);
  Py_RETURN_NONE;
}

static PyMethodDef eos_methods[] = {
  {"tabulated_enforce_bounds_rho_Ye_T",
   eos_tabulated_enforce_bounds_rho_Ye_T,
   METH_VARARGS,
   "Clip (rho, Ye, T) into EOS bounds."},
  {"tabulated_enforce_bounds_rho_Ye_eps",
   eos_tabulated_enforce_bounds_rho_Ye_eps,
   METH_VARARGS,
   "Clip (rho, Ye, eps) into EOS bounds."},
  {"tabulated_compute_P_from_T",
   eos_tabulated_compute_P_from_T,
   METH_VARARGS,
   "Compute pressure from (rho, Ye, T)."},
  {"tabulated_compute_eps_from_T",
   eos_tabulated_compute_eps_from_T,
   METH_VARARGS,
   "Compute specific internal energy from (rho, Ye, T)."},
  {"tabulated_compute_cs2_from_T",
   eos_tabulated_compute_cs2_from_T,
   METH_VARARGS,
   "Compute sound speed squared from (rho, Ye, T)."},
  {"tabulated_compute_P_eps_from_T",
   eos_tabulated_compute_P_eps_from_T,
   METH_VARARGS,
   "Compute (P, eps) from (rho, Ye, T)."},
  {"tabulated_compute_P_eps_S_from_T",
   eos_tabulated_compute_P_eps_S_from_T,
   METH_VARARGS,
   "Compute (P, eps, entropy) from (rho, Ye, T)."},
  {"tabulated_compute_T_from_eps",
   eos_tabulated_compute_T_from_eps,
   METH_VARARGS,
   "Compute T from (rho, Ye, eps)."},
  {"tabulated_compute_P_T_from_eps",
   eos_tabulated_compute_P_T_from_eps,
   METH_VARARGS,
   "Compute (P, T) from (rho, Ye, eps)."},
  {"load_nn_c2p_hdf5",
   eos_load_nn_c2p_hdf5,
   METH_VARARGS,
   "Load a Palenzuela1D NN initial-guess model from HDF5."},
  {"close", eos_close, METH_NOARGS, "Free EOS table memory early."},
  {NULL, NULL, 0, NULL}
};

static PyGetSetDef eos_getset[] = {
  {"rho_min", eos_get_rho_min, NULL, "Minimum density bound.", NULL},
  {"rho_max", eos_get_rho_max, NULL, "Maximum density bound.", NULL},
  {"Ye_min", eos_get_ye_min, NULL, "Minimum electron-fraction bound.", NULL},
  {"Ye_max", eos_get_ye_max, NULL, "Maximum electron-fraction bound.", NULL},
  {"T_min", eos_get_t_min, NULL, "Minimum temperature bound.", NULL},
  {"T_max", eos_get_t_max, NULL, "Maximum temperature bound.", NULL},
  {"table_T_min", eos_get_table_t_min, NULL, "Minimum table temperature.", NULL},
  {"table_T_max", eos_get_table_t_max, NULL, "Maximum table temperature.", NULL},
  {"root_finding_precision",
   eos_get_root_finding_precision,
   eos_set_root_finding_precision,
   "Root-finding precision for table inversions.",
   NULL},
  {"enable_neural_net_c2p",
   eos_get_enable_neural_net_c2p,
   eos_set_enable_neural_net_c2p,
   "Whether tabulated Con2Prim should use the embedded neural-network initial guess.",
   NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyGHLTabulatedEOSType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.TabulatedEOS",
  .tp_basicsize = sizeof(PyGHLTabulatedEOS),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Tabulated EOS wrapper with owned GRHayL table memory.",
  .tp_dealloc = eos_dealloc,
  .tp_repr = eos_repr,
  .tp_methods = eos_methods,
  .tp_getset = eos_getset,
};

DEFINE_DOUBLE_GETTER(primitive_get_rho, PyGHLPrimitive, prims.rho)
DEFINE_DOUBLE_SETTER(primitive_set_rho, PyGHLPrimitive, prims.rho, "rho")
DEFINE_DOUBLE_GETTER(primitive_get_press, PyGHLPrimitive, prims.press)
DEFINE_DOUBLE_SETTER(primitive_set_press, PyGHLPrimitive, prims.press, "press")
DEFINE_DOUBLE_GETTER(primitive_get_eps, PyGHLPrimitive, prims.eps)
DEFINE_DOUBLE_SETTER(primitive_set_eps, PyGHLPrimitive, prims.eps, "eps")
DEFINE_DOUBLE_GETTER(primitive_get_u0, PyGHLPrimitive, prims.u0)
DEFINE_DOUBLE_SETTER(primitive_set_u0, PyGHLPrimitive, prims.u0, "u0")
DEFINE_DOUBLE_GETTER(primitive_get_Ye, PyGHLPrimitive, prims.Y_e)
DEFINE_DOUBLE_SETTER(primitive_set_Ye, PyGHLPrimitive, prims.Y_e, "Y_e")
DEFINE_DOUBLE_GETTER(primitive_get_temperature, PyGHLPrimitive, prims.temperature)
DEFINE_DOUBLE_SETTER(
      primitive_set_temperature,
      PyGHLPrimitive,
      prims.temperature,
      "temperature")
DEFINE_DOUBLE_GETTER(primitive_get_entropy, PyGHLPrimitive, prims.entropy)
DEFINE_DOUBLE_SETTER(primitive_set_entropy, PyGHLPrimitive, prims.entropy, "entropy")

static PyObject *primitive_get_vU(PyObject *self, void *closure) {
  (void)closure;
  return build_vector3(((PyGHLPrimitive *)self)->prims.vU);
}

static int primitive_set_vU(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  if(value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete vU.");
    return -1;
  }
  return parse_vector3(value, ((PyGHLPrimitive *)self)->prims.vU, "vU");
}

static PyObject *primitive_get_BU(PyObject *self, void *closure) {
  (void)closure;
  return build_vector3(((PyGHLPrimitive *)self)->prims.BU);
}

static int primitive_set_BU(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  if(value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete BU.");
    return -1;
  }
  return parse_vector3(value, ((PyGHLPrimitive *)self)->prims.BU, "BU");
}

static PyObject *primitive_repr(PyObject *self) {
  const PyGHLPrimitive *prims = (const PyGHLPrimitive *)self;
  char buffer[160];
  snprintf(
        buffer,
        sizeof(buffer),
        "Primitive(rho=%g, press=%g, eps=%g, u0=%g, Y_e=%g, temperature=%g)",
        prims->prims.rho,
        prims->prims.press,
        prims->prims.eps,
        prims->prims.u0,
        prims->prims.Y_e,
        prims->prims.temperature);
  return PyUnicode_FromString(buffer);
}

static PyGetSetDef primitive_getset[] = {
  {"rho", primitive_get_rho, primitive_set_rho, "Density.", NULL},
  {"press", primitive_get_press, primitive_set_press, "Pressure.", NULL},
  {"eps", primitive_get_eps, primitive_set_eps, "Specific internal energy.", NULL},
  {"u0", primitive_get_u0, primitive_set_u0, "Lorentz factor.", NULL},
  {"vU", primitive_get_vU, primitive_set_vU, "Spatial velocity.", NULL},
  {"BU", primitive_get_BU, primitive_set_BU, "Magnetic field.", NULL},
  {"Y_e", primitive_get_Ye, primitive_set_Ye, "Electron fraction.", NULL},
  {"temperature",
   primitive_get_temperature,
   primitive_set_temperature,
   "Temperature.",
   NULL},
  {"entropy", primitive_get_entropy, primitive_set_entropy, "Entropy.", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyGHLPrimitiveType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.Primitive",
  .tp_basicsize = sizeof(PyGHLPrimitive),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "GRHayL primitive variables.",
  .tp_repr = primitive_repr,
  .tp_getset = primitive_getset,
  .tp_new = PyType_GenericNew,
};

DEFINE_DOUBLE_GETTER(conservative_get_rho, PyGHLConservative, cons.rho)
DEFINE_DOUBLE_SETTER(conservative_set_rho, PyGHLConservative, cons.rho, "rho")
DEFINE_DOUBLE_GETTER(conservative_get_tau, PyGHLConservative, cons.tau)
DEFINE_DOUBLE_SETTER(conservative_set_tau, PyGHLConservative, cons.tau, "tau")
DEFINE_DOUBLE_GETTER(conservative_get_Ye, PyGHLConservative, cons.Y_e)
DEFINE_DOUBLE_SETTER(conservative_set_Ye, PyGHLConservative, cons.Y_e, "Y_e")
DEFINE_DOUBLE_GETTER(conservative_get_entropy, PyGHLConservative, cons.entropy)
DEFINE_DOUBLE_SETTER(
      conservative_set_entropy,
      PyGHLConservative,
      cons.entropy,
      "entropy")

static PyObject *conservative_get_SD(PyObject *self, void *closure) {
  (void)closure;
  return build_vector3(((PyGHLConservative *)self)->cons.SD);
}

static int conservative_set_SD(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  if(value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete SD.");
    return -1;
  }
  return parse_vector3(value, ((PyGHLConservative *)self)->cons.SD, "SD");
}

static PyObject *conservative_repr(PyObject *self) {
  const PyGHLConservative *cons = (const PyGHLConservative *)self;
  char buffer[128];
  snprintf(
        buffer,
        sizeof(buffer),
        "Conservative(rho=%g, tau=%g, Y_e=%g)",
        cons->cons.rho,
        cons->cons.tau,
        cons->cons.Y_e);
  return PyUnicode_FromString(buffer);
}

static PyGetSetDef conservative_getset[] = {
  {"rho", conservative_get_rho, conservative_set_rho, "Densitized density.", NULL},
  {"tau", conservative_get_tau, conservative_set_tau, "Energy density.", NULL},
  {"Y_e", conservative_get_Ye, conservative_set_Ye, "Densitized electron fraction.", NULL},
  {"SD", conservative_get_SD, conservative_set_SD, "Momentum density.", NULL},
  {"entropy",
   conservative_get_entropy,
   conservative_set_entropy,
   "Densitized entropy.",
   NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyGHLConservativeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.Conservative",
  .tp_basicsize = sizeof(PyGHLConservative),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "GRHayL conservative variables.",
  .tp_repr = conservative_repr,
  .tp_getset = conservative_getset,
  .tp_new = PyType_GenericNew,
};

DEFINE_DOUBLE_GETTER(metric_get_lapse, PyGHLMetric, metric.lapse)
DEFINE_DOUBLE_GETTER(metric_get_lapseinv, PyGHLMetric, metric.lapseinv)
DEFINE_DOUBLE_GETTER(metric_get_detgamma, PyGHLMetric, metric.detgamma)
DEFINE_DOUBLE_GETTER(metric_get_sqrt_detgamma, PyGHLMetric, metric.sqrt_detgamma)

static PyObject *metric_get_betaU(PyObject *self, void *closure) {
  (void)closure;
  return build_vector3(((PyGHLMetric *)self)->metric.betaU);
}

static PyObject *metric_get_gammaDD(PyObject *self, void *closure) {
  (void)closure;
  return build_matrix3x3(((PyGHLMetric *)self)->metric.gammaDD);
}

static PyObject *metric_get_gammaUU(PyObject *self, void *closure) {
  (void)closure;
  return build_matrix3x3(((PyGHLMetric *)self)->metric.gammaUU);
}

static PyObject *metric_repr(PyObject *self) {
  const PyGHLMetric *metric = (const PyGHLMetric *)self;
  char buffer[128];
  snprintf(
        buffer,
        sizeof(buffer),
        "Metric(lapse=%g, detgamma=%g, sqrt_detgamma=%g)",
        metric->metric.lapse,
        metric->metric.detgamma,
        metric->metric.sqrt_detgamma);
  return PyUnicode_FromString(buffer);
}

static PyGetSetDef metric_getset[] = {
  {"lapse", metric_get_lapse, NULL, "Lapse.", NULL},
  {"lapseinv", metric_get_lapseinv, NULL, "Inverse lapse.", NULL},
  {"detgamma", metric_get_detgamma, NULL, "det(gamma_ij).", NULL},
  {"sqrt_detgamma", metric_get_sqrt_detgamma, NULL, "sqrt(det(gamma_ij)).", NULL},
  {"betaU", metric_get_betaU, NULL, "Shift vector.", NULL},
  {"gammaDD", metric_get_gammaDD, NULL, "Spatial metric.", NULL},
  {"gammaUU", metric_get_gammaUU, NULL, "Inverse spatial metric.", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyGHLMetricType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.Metric",
  .tp_basicsize = sizeof(PyGHLMetric),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "GRHayL ADM metric quantities.",
  .tp_repr = metric_repr,
  .tp_getset = metric_getset,
  .tp_new = PyType_GenericNew,
};

static PyObject *adm_aux_repr(PyObject *self) {
  (void)self;
  return PyUnicode_FromString("ADMAux()");
}

static PyTypeObject PyGHLADMAuxType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.ADMAux",
  .tp_basicsize = sizeof(PyGHLADMAux),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "GRHayL ADM auxiliary quantities.",
  .tp_repr = adm_aux_repr,
  .tp_new = PyType_GenericNew,
};

DEFINE_BOOL_GETTER(diagnostics_get_tau_fix, PyGHLDiagnostics, diagnostics.tau_fix)
DEFINE_BOOL_SETTER(diagnostics_set_tau_fix, PyGHLDiagnostics, diagnostics.tau_fix, "tau_fix")
DEFINE_BOOL_GETTER(
      diagnostics_get_Stilde_fix,
      PyGHLDiagnostics,
      diagnostics.Stilde_fix)
DEFINE_BOOL_SETTER(
      diagnostics_set_Stilde_fix,
      PyGHLDiagnostics,
      diagnostics.Stilde_fix,
      "Stilde_fix")
DEFINE_BOOL_GETTER(
      diagnostics_get_speed_limited,
      PyGHLDiagnostics,
      diagnostics.speed_limited)
DEFINE_BOOL_SETTER(
      diagnostics_set_speed_limited,
      PyGHLDiagnostics,
      diagnostics.speed_limited,
      "speed_limited")
static PyObject *diagnostics_get_which_routine(PyObject *self, void *closure) {
  (void)closure;
  return PyLong_FromLong(((PyGHLDiagnostics *)self)->diagnostics.which_routine);
}

static int diagnostics_set_which_routine(PyObject *self, PyObject *value, void *closure) {
  long parsed = 0;
  (void)closure;
  if(value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete which_routine.");
    return -1;
  }
  parsed = PyLong_AsLong(value);
  if(PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "which_routine must be an integer.");
    return -1;
  }
  ((PyGHLDiagnostics *)self)->diagnostics.which_routine = (ghl_con2prim_id_t)parsed;
  return 0;
}

static PyObject *diagnostics_get_n_iter(PyObject *self, void *closure) {
  (void)closure;
  return PyLong_FromLong(((PyGHLDiagnostics *)self)->diagnostics.n_iter);
}

static int diagnostics_set_n_iter(PyObject *self, PyObject *value, void *closure) {
  long parsed = 0;
  (void)closure;
  if(value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete n_iter.");
    return -1;
  }
  parsed = PyLong_AsLong(value);
  if(PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "n_iter must be an integer.");
    return -1;
  }
  ((PyGHLDiagnostics *)self)->diagnostics.n_iter = (int)parsed;
  return 0;
}

static PyObject *diagnostics_get_backup(PyObject *self, void *closure) {
  (void)closure;
  const PyGHLDiagnostics *diagnostics = (const PyGHLDiagnostics *)self;
  return Py_BuildValue(
        "(OOO)",
        diagnostics->diagnostics.backup[0] ? Py_True : Py_False,
        diagnostics->diagnostics.backup[1] ? Py_True : Py_False,
        diagnostics->diagnostics.backup[2] ? Py_True : Py_False);
}

static PyObject *diagnostics_repr(PyObject *self) {
  const PyGHLDiagnostics *diagnostics = (const PyGHLDiagnostics *)self;
  char buffer[160];
  snprintf(
        buffer,
        sizeof(buffer),
        "Diagnostics(n_iter=%d, speed_limited=%s, which_routine=%d)",
        diagnostics->diagnostics.n_iter,
        diagnostics->diagnostics.speed_limited ? "True" : "False",
        diagnostics->diagnostics.which_routine);
  return PyUnicode_FromString(buffer);
}

static PyGetSetDef diagnostics_getset[] = {
  {"tau_fix", diagnostics_get_tau_fix, diagnostics_set_tau_fix, "Whether tau was fixed.", NULL},
  {"Stilde_fix",
   diagnostics_get_Stilde_fix,
   diagnostics_set_Stilde_fix,
   "Whether S~ was fixed.",
   NULL},
  {"speed_limited",
   diagnostics_get_speed_limited,
   diagnostics_set_speed_limited,
   "Whether speed was limited.",
   NULL},
  {"which_routine",
   diagnostics_get_which_routine,
   diagnostics_set_which_routine,
   "Con2Prim routine used.",
   NULL},
  {"backup", diagnostics_get_backup, NULL, "Tuple indicating backup usage.", NULL},
  {"n_iter", diagnostics_get_n_iter, diagnostics_set_n_iter, "Iteration count.", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyGHLDiagnosticsType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pyghl.Diagnostics",
  .tp_basicsize = sizeof(PyGHLDiagnostics),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "GRHayL con2prim diagnostics.",
  .tp_repr = diagnostics_repr,
  .tp_getset = diagnostics_getset,
  .tp_new = diagnostics_new,
};

static void initialize_py_diagnostics(PyGHLDiagnostics *diagnostics) {
  memset(&diagnostics->diagnostics, 0, sizeof(diagnostics->diagnostics));
  ghl_initialize_diagnostics(&diagnostics->diagnostics);
}

static PyObject *diagnostics_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  PyGHLDiagnostics *diagnostics = (PyGHLDiagnostics *)type->tp_alloc(type, 0);
  if(diagnostics != NULL) {
    initialize_py_diagnostics(diagnostics);
  }
  return (PyObject *)diagnostics;
}

static int require_type(PyObject *obj, PyTypeObject *type, const char *name) {
  if(PyObject_TypeCheck(obj, type)) {
    return 0;
  }
  PyErr_Format(PyExc_TypeError, "%s must be a %s object.", name, type->tp_name);
  return -1;
}

static PyObject *py_initialize_params(PyObject *module, PyObject *args, PyObject *kwargs) {
  (void)module;
  int main_routine = ghl_con2prim_id_None;
  PyObject *backup_routine = Py_None;
  int evolve_entropy = 0;
  int evolve_temp = 1;
  int calc_prim_guess = 1;
  double psi6threshold = 1e100;
  double max_lorentz_factor = 100.0;
  double lorenz_damping_factor = 0.0;

  static char *kwlist[] = {
        "main_routine",
        "backup_routine",
        "evolve_entropy",
        "evolve_temp",
        "calc_prim_guess",
        "psi6threshold",
        "max_lorentz_factor",
        "lorenz_damping_factor",
        NULL};

  if(!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|iOpppddd:initialize_params",
            kwlist,
            &main_routine,
            &backup_routine,
            &evolve_entropy,
            &evolve_temp,
            &calc_prim_guess,
            &psi6threshold,
            &max_lorentz_factor,
            &lorenz_damping_factor)) {
    return NULL;
  }

  ghl_con2prim_id_t backup_methods[3] = {
        ghl_con2prim_id_None,
        ghl_con2prim_id_None,
        ghl_con2prim_id_None};
  if(backup_routine != Py_None) {
    PyObject *sequence = PySequence_Fast(
          backup_routine,
          "backup_routine must be a 3-element sequence of integer method keys.");
    if(sequence == NULL) {
      return NULL;
    }

    if(PySequence_Fast_GET_SIZE(sequence) != 3) {
      Py_DECREF(sequence);
      PyErr_SetString(
            PyExc_ValueError,
            "backup_routine must contain exactly 3 method keys.");
      return NULL;
    }

    for(Py_ssize_t i = 0; i < 3; i++) {
      const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, i));
      if(PyErr_Occurred()) {
        Py_DECREF(sequence);
        return NULL;
      }
      backup_methods[i] = (ghl_con2prim_id_t)value;
    }
    Py_DECREF(sequence);
  }

  PyGHLParams *params = PyObject_New(PyGHLParams, &PyGHLParamsType);
  if(params == NULL) {
    return NULL;
  }
  memset(&params->params, 0, sizeof(params->params));

  ghl_initialize_params(
        (ghl_con2prim_id_t)main_routine,
        backup_methods,
        (bool)evolve_entropy,
        (bool)evolve_temp,
        (bool)calc_prim_guess,
        psi6threshold,
        max_lorentz_factor,
        lorenz_damping_factor,
        &params->params);

  return (PyObject *)params;
}

static PyObject *py_initialize_tabulated_eos_functions_and_params(
      PyObject *module,
      PyObject *args,
      PyObject *kwargs) {
  (void)module;

  PyObject *params_obj = NULL;
  const char *table_path = NULL;

  double rho_atm = 1e-12;
  double rho_min = 1e-12;
  double rho_max = 1e300;
  double Ye_atm = 0.5;
  double Ye_min = 0.05;
  double Ye_max = 0.5;
  double T_atm = 1e-2;
  double T_min = 1e-2;
  double T_max = 1e2;
  double root_finding_precision = 1e-10;
  int enable_neural_net_c2p = 0;

  static char *kwlist[] = {
        "params",
        "table_path",
        "rho_atm",
        "rho_min",
        "rho_max",
        "Ye_atm",
        "Ye_min",
        "Ye_max",
        "T_atm",
        "T_min",
        "T_max",
        "root_finding_precision",
        "enable_neural_net_c2p",
        NULL};

  if(!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Os|ddddddddddp:initialize_tabulated_eos_functions_and_params",
            kwlist,
            &params_obj,
            &table_path,
            &rho_atm,
            &rho_min,
            &rho_max,
            &Ye_atm,
            &Ye_min,
            &Ye_max,
            &T_atm,
            &T_min,
            &T_max,
            &root_finding_precision,
            &enable_neural_net_c2p)) {
    return NULL;
  }

  if(require_type(params_obj, &PyGHLParamsType, "params") < 0) {
    return NULL;
  }
  if(access(table_path, R_OK) != 0) {
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError, table_path);
  }

  PyGHLTabulatedEOS *eos = PyObject_New(PyGHLTabulatedEOS, &PyGHLTabulatedEOSType);
  if(eos == NULL) {
    return NULL;
  }

  memset(&eos->eos, 0, sizeof(eos->eos));
  eos->initialized = 0;
  eos->eos.enable_neural_net_c2p = (bool)enable_neural_net_c2p;

  ghl_initialize_tabulated_eos_functions_and_params(
        table_path,
        rho_atm,
        rho_min,
        rho_max,
        Ye_atm,
        Ye_min,
        Ye_max,
        T_atm,
        T_min,
        T_max,
        &eos->eos);

  eos->eos.root_finding_precision = root_finding_precision;
  eos->initialized = 1;

  return (PyObject *)eos;
}

static PyObject *py_initialize_metric(PyObject *module, PyObject *args) {
  (void)module;
  double lapse, betax, betay, betaz, gxx, gxy, gxz, gyy, gyz, gzz;
  if(!PyArg_ParseTuple(
            args,
            "dddddddddd:initialize_metric",
            &lapse,
            &betax,
            &betay,
            &betaz,
            &gxx,
            &gxy,
            &gxz,
            &gyy,
            &gyz,
            &gzz)) {
    return NULL;
  }

  PyGHLMetric *metric = PyObject_New(PyGHLMetric, &PyGHLMetricType);
  if(metric == NULL) {
    return NULL;
  }
  memset(&metric->metric, 0, sizeof(metric->metric));

  ghl_initialize_metric(
        lapse,
        betax,
        betay,
        betaz,
        gxx,
        gxy,
        gxz,
        gyy,
        gyz,
        gzz,
        &metric->metric);

  return (PyObject *)metric;
}

static PyObject *py_compute_ADM_auxiliaries(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *metric_obj = NULL;
  if(!PyArg_ParseTuple(args, "O:compute_ADM_auxiliaries", &metric_obj)) {
    return NULL;
  }
  if(require_type(metric_obj, &PyGHLMetricType, "metric") < 0) {
    return NULL;
  }

  PyGHLADMAux *aux = PyObject_New(PyGHLADMAux, &PyGHLADMAuxType);
  if(aux == NULL) {
    return NULL;
  }
  memset(&aux->aux, 0, sizeof(aux->aux));
  ghl_compute_ADM_auxiliaries(&((PyGHLMetric *)metric_obj)->metric, &aux->aux);
  return (PyObject *)aux;
}

static PyObject *py_initialize_primitives(PyObject *module, PyObject *args) {
  (void)module;
  double rho, press, eps, vx, vy, vz, Bx, By, Bz, entropy, Y_e, temperature;
  if(!PyArg_ParseTuple(
            args,
            "dddddddddddd:initialize_primitives",
            &rho,
            &press,
            &eps,
            &vx,
            &vy,
            &vz,
            &Bx,
            &By,
            &Bz,
            &entropy,
            &Y_e,
            &temperature)) {
    return NULL;
  }

  PyGHLPrimitive *prims = PyObject_New(PyGHLPrimitive, &PyGHLPrimitiveType);
  if(prims == NULL) {
    return NULL;
  }
  memset(&prims->prims, 0, sizeof(prims->prims));

  ghl_initialize_primitives(
        rho,
        press,
        eps,
        vx,
        vy,
        vz,
        Bx,
        By,
        Bz,
        entropy,
        Y_e,
        temperature,
        &prims->prims);

  return (PyObject *)prims;
}

static PyObject *py_initialize_diagnostics(PyObject *module, PyObject *args) {
  (void)module;
  if(!PyArg_ParseTuple(args, ":initialize_diagnostics")) {
    return NULL;
  }
  PyGHLDiagnostics *diagnostics = PyObject_New(PyGHLDiagnostics, &PyGHLDiagnosticsType);
  if(diagnostics == NULL) {
    return NULL;
  }
  initialize_py_diagnostics(diagnostics);
  return (PyObject *)diagnostics;
}

static PyObject *py_compute_conservs(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *metric_obj = NULL;
  PyObject *aux_obj = NULL;
  PyObject *prims_obj = NULL;
  if(!PyArg_ParseTuple(args, "OOO:compute_conservs", &metric_obj, &aux_obj, &prims_obj)) {
    return NULL;
  }
  if(require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(aux_obj, &PyGHLADMAuxType, "metric_aux") < 0
     || require_type(prims_obj, &PyGHLPrimitiveType, "prims") < 0) {
    return NULL;
  }

  PyGHLConservative *cons = PyObject_New(PyGHLConservative, &PyGHLConservativeType);
  if(cons == NULL) {
    return NULL;
  }
  memset(&cons->cons, 0, sizeof(cons->cons));
  ghl_compute_conservs(
        &((PyGHLMetric *)metric_obj)->metric,
        &((PyGHLADMAux *)aux_obj)->aux,
        &((PyGHLPrimitive *)prims_obj)->prims,
        &cons->cons);
  return (PyObject *)cons;
}

static PyObject *py_undensitize_conservatives(PyObject *module, PyObject *args) {
  (void)module;
  double psi6;
  PyObject *cons_obj = NULL;
  if(!PyArg_ParseTuple(args, "dO:undensitize_conservatives", &psi6, &cons_obj)) {
    return NULL;
  }
  if(require_type(cons_obj, &PyGHLConservativeType, "cons") < 0) {
    return NULL;
  }

  PyGHLConservative *cons_undens = PyObject_New(PyGHLConservative, &PyGHLConservativeType);
  if(cons_undens == NULL) {
    return NULL;
  }
  memset(&cons_undens->cons, 0, sizeof(cons_undens->cons));
  ghl_undensitize_conservatives(psi6, &((PyGHLConservative *)cons_obj)->cons, &cons_undens->cons);
  return (PyObject *)cons_undens;
}

static PyObject *py_compute_SU_Bsq_Ssq_BdotS(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *metric_obj = NULL;
  PyObject *cons_obj = NULL;
  PyObject *prims_obj = NULL;
  double SU[3];
  double B_squared;
  double S_squared;
  double BdotS;

  if(!PyArg_ParseTuple(
            args,
            "OOO:compute_SU_Bsq_Ssq_BdotS",
            &metric_obj,
            &cons_obj,
            &prims_obj)) {
    return NULL;
  }
  if(require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(cons_obj, &PyGHLConservativeType, "cons") < 0
     || require_type(prims_obj, &PyGHLPrimitiveType, "prims") < 0) {
    return NULL;
  }

  ghl_compute_SU_Bsq_Ssq_BdotS(
        &((PyGHLMetric *)metric_obj)->metric,
        &((PyGHLConservative *)cons_obj)->cons,
        &((PyGHLPrimitive *)prims_obj)->prims,
        SU,
        &B_squared,
        &S_squared,
        &BdotS);

  return Py_BuildValue("((ddd)ddd)", SU[0], SU[1], SU[2], B_squared, S_squared, BdotS);
}

static PyObject *py_limit_v_and_compute_u0(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *params_obj = NULL;
  PyObject *metric_obj = NULL;
  PyObject *prims_obj = NULL;
  bool speed_limited = false;

  if(!PyArg_ParseTuple(
            args,
            "OOO:limit_v_and_compute_u0",
            &params_obj,
            &metric_obj,
            &prims_obj)) {
    return NULL;
  }
  if(require_type(params_obj, &PyGHLParamsType, "params") < 0
     || require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(prims_obj, &PyGHLPrimitiveType, "prims") < 0) {
    return NULL;
  }

  const ghl_error_codes_t error = ghl_limit_v_and_compute_u0(
        &((PyGHLParams *)params_obj)->params,
        &((PyGHLMetric *)metric_obj)->metric,
        &((PyGHLPrimitive *)prims_obj)->prims,
        &speed_limited);
  if(error != ghl_success) {
    raise_ghl_error("limit_v_and_compute_u0", error);
    return NULL;
  }
  return PyBool_FromLong(speed_limited);
}

static PyObject *py_limit_utilde_and_compute_v(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *params_obj = NULL;
  PyObject *metric_obj = NULL;
  PyObject *utilde_obj = NULL;
  PyObject *prims_obj = NULL;
  double utildeU[3];
  bool speed_limited = false;

  if(!PyArg_ParseTuple(
            args,
            "OOOO:limit_utilde_and_compute_v",
            &params_obj,
            &metric_obj,
            &utilde_obj,
            &prims_obj)) {
    return NULL;
  }
  if(require_type(params_obj, &PyGHLParamsType, "params") < 0
     || require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(prims_obj, &PyGHLPrimitiveType, "prims") < 0) {
    return NULL;
  }
  if(parse_vector3(utilde_obj, utildeU, "utildeU") < 0) {
    return NULL;
  }

  speed_limited = ghl_limit_utilde_and_compute_v(
        &((PyGHLParams *)params_obj)->params,
        &((PyGHLMetric *)metric_obj)->metric,
        utildeU,
        &((PyGHLPrimitive *)prims_obj)->prims);
  return PyBool_FromLong(speed_limited);
}

static PyObject *py_guess_primitives(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *params_obj = NULL;
  PyObject *eos_obj = NULL;
  PyObject *metric_obj = NULL;
  PyObject *cons_obj = NULL;
  if(!PyArg_ParseTuple(args, "OOOO:guess_primitives", &params_obj, &eos_obj, &metric_obj, &cons_obj)) {
    return NULL;
  }
  if(require_type(params_obj, &PyGHLParamsType, "params") < 0
     || require_type(eos_obj, &PyGHLTabulatedEOSType, "eos") < 0
     || require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(cons_obj, &PyGHLConservativeType, "cons") < 0) {
    return NULL;
  }
  if(eos_ensure_initialized((PyGHLTabulatedEOS *)eos_obj) < 0) {
    return NULL;
  }

  PyGHLPrimitive *prims = PyObject_New(PyGHLPrimitive, &PyGHLPrimitiveType);
  if(prims == NULL) {
    return NULL;
  }
  memset(&prims->prims, 0, sizeof(prims->prims));
  ghl_guess_primitives(
        &((PyGHLParams *)params_obj)->params,
        &((PyGHLTabulatedEOS *)eos_obj)->eos,
        &((PyGHLMetric *)metric_obj)->metric,
        &((PyGHLConservative *)cons_obj)->cons,
        &prims->prims);
  return (PyObject *)prims;
}

static PyObject *py_tabulated_Palenzuela1D_energy(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *params_obj = NULL;
  PyObject *eos_obj = NULL;
  PyObject *metric_obj = NULL;
  PyObject *aux_obj = NULL;
  PyObject *cons_obj = NULL;
  PyObject *prims_obj = NULL;
  PyObject *diagnostics_obj = Py_None;
  PyGHLDiagnostics *diagnostics = NULL;
  int created_diagnostics = 0;

  if(!PyArg_ParseTuple(
            args,
            "OOOOOO|O:tabulated_Palenzuela1D_energy",
            &params_obj,
            &eos_obj,
            &metric_obj,
            &aux_obj,
            &cons_obj,
            &prims_obj,
            &diagnostics_obj)) {
    return NULL;
  }
  if(require_type(params_obj, &PyGHLParamsType, "params") < 0
     || require_type(eos_obj, &PyGHLTabulatedEOSType, "eos") < 0
     || require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(aux_obj, &PyGHLADMAuxType, "metric_aux") < 0
     || require_type(cons_obj, &PyGHLConservativeType, "cons") < 0
     || require_type(prims_obj, &PyGHLPrimitiveType, "prims") < 0) {
    return NULL;
  }
  if(eos_ensure_initialized((PyGHLTabulatedEOS *)eos_obj) < 0) {
    return NULL;
  }

  if(diagnostics_obj == Py_None) {
    diagnostics = PyObject_New(PyGHLDiagnostics, &PyGHLDiagnosticsType);
    if(diagnostics == NULL) {
      return NULL;
    }
    initialize_py_diagnostics(diagnostics);
    diagnostics_obj = (PyObject *)diagnostics;
    created_diagnostics = 1;
  }
  else {
    if(require_type(diagnostics_obj, &PyGHLDiagnosticsType, "diagnostics") < 0) {
      return NULL;
    }
    diagnostics = (PyGHLDiagnostics *)diagnostics_obj;
  }

  const ghl_error_codes_t error = ghl_tabulated_Palenzuela1D_energy(
        &((PyGHLParams *)params_obj)->params,
        &((PyGHLTabulatedEOS *)eos_obj)->eos,
        &((PyGHLMetric *)metric_obj)->metric,
        &((PyGHLADMAux *)aux_obj)->aux,
        &((PyGHLConservative *)cons_obj)->cons,
        &((PyGHLPrimitive *)prims_obj)->prims,
        &diagnostics->diagnostics);
  if(error != ghl_success) {
    if(created_diagnostics) {
      Py_DECREF(diagnostics_obj);
    }
    raise_ghl_error("tabulated_Palenzuela1D_energy", error);
    return NULL;
  }

  Py_INCREF(diagnostics_obj);
  return diagnostics_obj;
}

static PyObject *py_tabulated_con2prim_multi_method(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *params_obj = NULL;
  PyObject *eos_obj = NULL;
  PyObject *metric_obj = NULL;
  PyObject *aux_obj = NULL;
  PyObject *cons_obj = NULL;
  PyObject *prims_obj = NULL;
  PyObject *diagnostics_obj = Py_None;
  PyGHLDiagnostics *diagnostics = NULL;
  int created_diagnostics = 0;

  if(!PyArg_ParseTuple(
            args,
            "OOOOOO|O:tabulated_con2prim_multi_method",
            &params_obj,
            &eos_obj,
            &metric_obj,
            &aux_obj,
            &cons_obj,
            &prims_obj,
            &diagnostics_obj)) {
    return NULL;
  }
  if(require_type(params_obj, &PyGHLParamsType, "params") < 0
     || require_type(eos_obj, &PyGHLTabulatedEOSType, "eos") < 0
     || require_type(metric_obj, &PyGHLMetricType, "metric") < 0
     || require_type(aux_obj, &PyGHLADMAuxType, "metric_aux") < 0
     || require_type(cons_obj, &PyGHLConservativeType, "cons") < 0
     || require_type(prims_obj, &PyGHLPrimitiveType, "prims") < 0) {
    return NULL;
  }
  if(eos_ensure_initialized((PyGHLTabulatedEOS *)eos_obj) < 0) {
    return NULL;
  }

  if(diagnostics_obj == Py_None) {
    diagnostics = PyObject_New(PyGHLDiagnostics, &PyGHLDiagnosticsType);
    if(diagnostics == NULL) {
      return NULL;
    }
    initialize_py_diagnostics(diagnostics);
    diagnostics_obj = (PyObject *)diagnostics;
    created_diagnostics = 1;
  }
  else {
    if(require_type(diagnostics_obj, &PyGHLDiagnosticsType, "diagnostics") < 0) {
      return NULL;
    }
    diagnostics = (PyGHLDiagnostics *)diagnostics_obj;
  }

  const ghl_error_codes_t error = ghl_con2prim_tabulated_multi_method(
        &((PyGHLParams *)params_obj)->params,
        &((PyGHLTabulatedEOS *)eos_obj)->eos,
        &((PyGHLMetric *)metric_obj)->metric,
        &((PyGHLADMAux *)aux_obj)->aux,
        &((PyGHLConservative *)cons_obj)->cons,
        &((PyGHLPrimitive *)prims_obj)->prims,
        &diagnostics->diagnostics);
  if(error != ghl_success) {
    if(created_diagnostics) {
      Py_DECREF(diagnostics_obj);
    }
    raise_ghl_error("tabulated_con2prim_multi_method", error);
    return NULL;
  }

  Py_INCREF(diagnostics_obj);
  return diagnostics_obj;
}

static PyObject *py_nn_c2p_guess(PyObject *module, PyObject *args) {
  (void)module;
  PyObject *eos_obj = NULL;
  float q = 0.0f;
  float r = 0.0f;
  float s = 0.0f;
  float t = 0.0f;
  if(!PyArg_ParseTuple(args, "Offff:nn_c2p_guess", &eos_obj, &q, &r, &s, &t)) {
    return NULL;
  }
  if(require_type(eos_obj, &PyGHLTabulatedEOSType, "eos") < 0) {
    return NULL;
  }
  PyGHLTabulatedEOS *eos = (PyGHLTabulatedEOS *)eos_obj;
  if(eos_ensure_initialized(eos) < 0) {
    return NULL;
  }
  if(eos->eos.c2p_nn == NULL) {
    PyErr_SetString(GRHayLError, "eos has no loaded nn_c2p model; call eos.load_nn_c2p_hdf5(...) first");
    return NULL;
  }
  const ghl_nn_c2p_input_t input = { q, r, s, t };
  const ghl_nn_c2p_guess_t guess = ghl_c2p_nn_guess(eos->eos.c2p_nn, input);
  return PyFloat_FromDouble((double)guess.x);
}

static PyObject *py_nn_c2p_guess_x(PyObject *module, PyObject *args) {
  return py_nn_c2p_guess(module, args);
}

static PyMethodDef module_methods[] = {
  {"initialize_params",
   (PyCFunction)py_initialize_params,
   METH_VARARGS | METH_KEYWORDS,
   "Initialize GRHayL params with sensible defaults."},
  {"initialize_tabulated_eos_functions_and_params",
   (PyCFunction)py_initialize_tabulated_eos_functions_and_params,
   METH_VARARGS | METH_KEYWORDS,
   "Initialize a tabulated EOS object and associated function pointers."},
  {"initialize_metric", py_initialize_metric, METH_VARARGS, "Initialize ADM metric values."},
  {"compute_ADM_auxiliaries",
   py_compute_ADM_auxiliaries,
   METH_VARARGS,
   "Compute ADM auxiliary metric quantities."},
  {"initialize_primitives",
   py_initialize_primitives,
   METH_VARARGS,
   "Initialize primitive variables."},
  {"initialize_diagnostics",
   py_initialize_diagnostics,
   METH_VARARGS,
   "Initialize diagnostics."},
  {"compute_conservs", py_compute_conservs, METH_VARARGS, "Compute conservative variables."},
  {"undensitize_conservatives",
   py_undensitize_conservatives,
   METH_VARARGS,
   "Undensitize conservative variables."},
  {"compute_SU_Bsq_Ssq_BdotS",
   py_compute_SU_Bsq_Ssq_BdotS,
   METH_VARARGS,
   "Compute Palenzuela auxiliary quantities."},
  {"limit_v_and_compute_u0",
   py_limit_v_and_compute_u0,
   METH_VARARGS,
   "Limit velocity and compute u0."},
  {"limit_utilde_and_compute_v",
   py_limit_utilde_and_compute_v,
   METH_VARARGS,
   "Limit utilde and compute velocity."},
  {"guess_primitives", py_guess_primitives, METH_VARARGS, "Compute primitive guesses."},
  {"tabulated_Palenzuela1D_energy",
   py_tabulated_Palenzuela1D_energy,
   METH_VARARGS,
   "Run tabulated Palenzuela1D energy con2prim."},
  {"tabulated_con2prim_multi_method",
   py_tabulated_con2prim_multi_method,
   METH_VARARGS,
   "Run tabulated con2prim using params.main_routine and current primitive guess."},
  {"nn_c2p_guess", py_nn_c2p_guess, METH_VARARGS, "Predict x with the embedded NN."},
  {"nn_c2p_guess_x", py_nn_c2p_guess_x, METH_VARARGS, "Predict x with the embedded NN."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pyghl_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "_pyghl",
  .m_doc = "Low-level Python bindings for GRHayL.",
  .m_size = -1,
  .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__pyghl(void) {
  if(PyType_Ready(&PyGHLParamsType) < 0 || PyType_Ready(&PyGHLTabulatedEOSType) < 0
     || PyType_Ready(&PyGHLPrimitiveType) < 0 || PyType_Ready(&PyGHLConservativeType) < 0
     || PyType_Ready(&PyGHLMetricType) < 0 || PyType_Ready(&PyGHLADMAuxType) < 0
     || PyType_Ready(&PyGHLDiagnosticsType) < 0) {
    return NULL;
  }

  PyObject *module = PyModule_Create(&pyghl_module);
  if(module == NULL) {
    return NULL;
  }

  GRHayLError = PyErr_NewException("pyghl.GRHayLError", PyExc_RuntimeError, NULL);
  if(GRHayLError == NULL) {
    Py_DECREF(module);
    return NULL;
  }
  if(PyModule_AddObject(module, "GRHayLError", GRHayLError) < 0) {
    Py_DECREF(GRHayLError);
    Py_DECREF(module);
    return NULL;
  }

  Py_INCREF(&PyGHLParamsType);
  PyModule_AddObject(module, "Params", (PyObject *)&PyGHLParamsType);
  Py_INCREF(&PyGHLTabulatedEOSType);
  PyModule_AddObject(module, "TabulatedEOS", (PyObject *)&PyGHLTabulatedEOSType);
  Py_INCREF(&PyGHLPrimitiveType);
  PyModule_AddObject(module, "Primitive", (PyObject *)&PyGHLPrimitiveType);
  Py_INCREF(&PyGHLConservativeType);
  PyModule_AddObject(module, "Conservative", (PyObject *)&PyGHLConservativeType);
  Py_INCREF(&PyGHLMetricType);
  PyModule_AddObject(module, "Metric", (PyObject *)&PyGHLMetricType);
  Py_INCREF(&PyGHLADMAuxType);
  PyModule_AddObject(module, "ADMAux", (PyObject *)&PyGHLADMAuxType);
  Py_INCREF(&PyGHLDiagnosticsType);
  PyModule_AddObject(module, "Diagnostics", (PyObject *)&PyGHLDiagnosticsType);

  if(PyModule_AddIntConstant(module, "C2P_NONE", ghl_con2prim_id_None) < 0
     || PyModule_AddIntConstant(module, "C2P_NOBLE2D", ghl_con2prim_id_Noble2D) < 0
     || PyModule_AddIntConstant(module, "C2P_NOBLE1D", ghl_con2prim_id_Noble1D) < 0
     || PyModule_AddIntConstant(
              module,
              "C2P_NOBLE1D_ENTROPY",
              ghl_con2prim_id_Noble1D_entropy)
           < 0
     || PyModule_AddIntConstant(
              module,
              "C2P_NOBLE1D_ENTROPY2",
              ghl_con2prim_id_Noble1D_entropy2)
           < 0
     || PyModule_AddIntConstant(module, "C2P_FONT1D", ghl_con2prim_id_Font1D) < 0
     || PyModule_AddIntConstant(
              module,
              "C2P_PALENZUELA1D",
              ghl_con2prim_id_Palenzuela1D)
           < 0
     || PyModule_AddIntConstant(
              module,
              "C2P_PALENZUELA1D_ENTROPY",
              ghl_con2prim_id_Palenzuela1D_entropy)
           < 0
     || PyModule_AddIntConstant(module, "C2P_NEWMAN1D", ghl_con2prim_id_Newman1D) < 0
     || PyModule_AddIntConstant(
              module,
              "C2P_NEWMAN1D_ENTROPY",
              ghl_con2prim_id_Newman1D_entropy)
           < 0) {
    Py_DECREF(module);
    return NULL;
  }

  return module;
}
