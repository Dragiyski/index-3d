#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <functional>
#include <memory>
#include <numpy/arrayobject.h>
#include <set>
#include <vector>

PyObject* floatlist_create(PyObject* self, PyObject *args);

static PyMethodDef floatlist_methods[] = {
    { "create", floatlist_create, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef floatlist_module = {
    PyModuleDef_HEAD_INIT,
    "floatlist",
    NULL,
    -1,
    floatlist_methods
};

PyMODINIT_FUNC PyInit_floatlist() {
    PyObject* module;

    module = PyModule_Create(&floatlist_module);
    import_array1(0);

    return module;
}

typedef struct
{
    PyArrayObject* list;
    PyArrayObject* index;
} FloatListResult;

template <typename T>
struct float_is_equal : public std::binary_function<T, T, bool> {
    bool operator()(T a, T b) const {
        static const auto epsilon = std::numeric_limits<T>::epsilon();
        static const auto min_value = std::numeric_limits<T>::min();
        static const auto max_value = std::numeric_limits<T>::max();
        // Nothing is equal to NaN (not even NaN)
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        // If both are exactly equal, they are equal, no further checks.
        if (a == b) {
            return true;
        }
        auto difference = std::abs(a - b);
        auto finite_amplitude = std::min(std::abs(a) + std::abs(b), max_value);
        auto min_difference = std::max(min_value, finite_amplitude * epsilon);
        return difference < min_difference;
    }
};

template <typename T>
struct float_compare_less : public std::binary_function<T, T, bool> {
    bool operator()(T a, T b) const {
        static const auto is_equal_function = float_is_equal<T>();
        if (!is_equal_function(a, b)) {
            return a < b;
        }
        return false;
    }
};

template<typename T>
struct floatlist_numpy_type;

template<>
struct floatlist_numpy_type<float> {
    static const constexpr auto value = NPY_FLOAT32;
};

template<>
struct floatlist_numpy_type<double> {
    static const constexpr auto value = NPY_FLOAT64;
};

template<typename T>
bool floatlist_from_array(PyArrayObject* input, PyObject** output_value, PyObject** output_index) {
    auto input_ndim = PyArray_NDIM(input);
    auto input_shape = PyArray_SHAPE(input);
    static const auto max_index = std::numeric_limits<uint32_t>::max();
    std::set<T, float_compare_less<T>> float_list;
    static const auto iterator_deleter = [](NpyIter* iterator) {
        if (iterator != NULL) {
            NpyIter_Deallocate(iterator);
        }
    };
    {
        std::unique_ptr<NpyIter, decltype(iterator_deleter)> iterator(NpyIter_New(input, NPY_ITER_READONLY | NPY_ITER_ZEROSIZE_OK, NPY_KEEPORDER, NPY_NO_CASTING, NULL), iterator_deleter);
        if (!iterator) {
            return false;
        }
        auto iterator_size = NpyIter_GetIterSize(iterator.get());
        if (iterator_size <= 0) {
            // Empty iteration means empty output
            iterator_size = 0;
            *output_value = PyArray_SimpleNew(1, &iterator_size, floatlist_numpy_type<T>::value);
            *output_index = PyArray_SimpleNew(input_ndim, input_shape, NPY_UINT32);
            return true;
        }
        auto ptr_data = NpyIter_GetDataPtrArray(iterator.get());
        if (ptr_data == NULL) {
            return false;
        }
        auto next_iteration = NpyIter_GetIterNext(iterator.get(), NULL);
        if (next_iteration == NULL) {
            return false;
        }
        do {
            auto value = *reinterpret_cast<T*>(*ptr_data);
            auto insert_result = float_list.insert(value);
            if (insert_result.second && float_list.size() > max_index) [[unlikely]] {
                PyErr_SetString(PyExc_IndexError, "Float list exceeds u32 size");
                return false;
            }
        } while (next_iteration(iterator.get()));
    }
    {
        std::unique_ptr<NpyIter, decltype(iterator_deleter)> iterator(NpyIter_New(input, NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX | NPY_ITER_C_INDEX, NPY_KEEPORDER, NPY_NO_CASTING, NULL), iterator_deleter);
        if (!iterator) {
            return false;
        }
        auto ptr_data = NpyIter_GetDataPtrArray(iterator.get());
        if (ptr_data == NULL) {
            return false;
        }
        auto next_iteration = NpyIter_GetIterNext(iterator.get(), NULL);
        if (next_iteration == NULL) {
            return false;
        }
        auto get_multi_index = NpyIter_GetGetMultiIndex(iterator.get(), NULL);
        if (get_multi_index == NULL) {
            return false;
        }
        auto target_index_array = (PyArrayObject *) PyArray_New(&PyArray_Type, input_ndim, input_shape, NPY_UINT32, NULL, NULL, 0, NPY_ARRAY_OUT_ARRAY | NPY_ARRAY_ENSUREARRAY, NULL);
        if (target_index_array == NULL) {
            return false;
        }
        auto target_index_stride = PyArray_STRIDES(target_index_array);
        if (target_index_stride == NULL) {
            return false;
        }
        auto target_index_data = PyArray_DATA(target_index_array);
        if (target_index_data == NULL) {
            return false;
        }
        do {
            auto value = *reinterpret_cast<T*>(*ptr_data);
            auto location = float_list.find(value);
            if (location == float_list.end()) [[unlikely]] {
                PyErr_SetString(PyExc_RuntimeError, "Unable to find the floating-point value in the search phase. The value should have been inserted in the insert phase.");
                return false;
            }
            auto float_list_index = std::distance(float_list.begin(), location);
            if (float_list_index > std::numeric_limits<uint32_t>::max()) {
                PyErr_SetString(PyExc_IndexError, "Float list index exceeds u32 size");
                return false;
            }
            npy_intp multi_index[input_ndim];
            get_multi_index(iterator.get(), multi_index);
            auto target = target_index_data;
            for (decltype(input_ndim) i = 0; i < input_ndim; ++i) {
                target += multi_index[i] * target_index_stride[i];
            }
            *reinterpret_cast<uint32_t*>(target) = static_cast<uint32_t>(float_list_index);
        } while (next_iteration(iterator.get()));
        *output_index = (PyObject *)target_index_array;
    }
    {
        npy_intp value_size = float_list.size();
        auto value_array = (PyArrayObject *) PyArray_New(&PyArray_Type, 1, &value_size, floatlist_numpy_type<T>::value, NULL, NULL, 0, NPY_ARRAY_OUT_ARRAY | NPY_ARRAY_ENSUREARRAY, NULL);
        if (value_array == NULL) {
            return false;
        }
        auto stride = PyArray_STRIDE(value_array, 0);
        auto data = PyArray_DATA(value_array);
        if (data == NULL) {
            return false;
        }
        npy_intp i = 0;
        for (auto it = float_list.begin(); it != float_list.end(); ++it, ++i) {
            *reinterpret_cast<T *>(data + i * stride) = *it;
        }
        *output_value = (PyObject *)value_array;
    }
    return true;
}

PyObject* floatlist_create(PyObject* self, PyObject* args) {
    PyArrayObject *input;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input)) {
        return NULL;
    }
    PyObject *output_index, *output_data;
    if (PyArray_TYPE(input) == NPY_FLOAT32) {
        if (!floatlist_from_array<float>(input, &output_data, &output_index)) {
            return NULL;
        }
    } else if (PyArray_TYPE(input) == NPY_FLOAT64) {
        if (!floatlist_from_array<double>(input, &output_data, &output_index)) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "floatlist.from() unsupported NDArray dtype");
        return NULL;
    }
    return PyTuple_Pack(2, output_data, output_index);
}