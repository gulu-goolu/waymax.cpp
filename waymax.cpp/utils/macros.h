#pragma once

#include "absl/status/statusor.h"
#include <utility>

namespace waymax_cpp {
template <typename Fn> inline auto on_scope_exited(Fn &&fn) -> decltype(auto) {
  struct Helper {
    Helper(Fn &&_fn) : fn(std::move(_fn)) {}

    ~Helper() { fn(); }

    Fn fn;
  };

  return Helper(std::forward<Fn>(fn));
}

} // namespace waymax_cpp

#define COMMON_AS_STRING(x) COMMON_AS_STRING2(x)
#define COMMON_AS_STRING2(x) #x

#define COMMON_DEBUG_LOCATION __FILE__ ":" COMMON_AS_STRING(__LINE__)

// use COMMON_RETURN_IF_FAILED instead of this macro
#define COMMON_RETURN_IF_FAILED_1(expr)                                        \
  do {                                                                         \
    auto st = (expr);                                                          \
    if (!st.ok()) {                                                            \
      return absl::Status(st.code(), COMMON_DEBUG_LOCATION " " #expr           \
                                                           " failed\n" +       \
                                         st.ToString());                       \
    }                                                                          \
  } while (false)

// use COMMON_RETURN_IF_FAILED instead of this macro
#define COMMON_RETURN_IF_FAILED_2(expr, MSG)                                   \
  do {                                                                         \
    auto st = (expr);                                                          \
    if (!st.ok()) {                                                            \
      return absl::Status(st.code(), COMMON_DEBUG_LOCATION " " #expr           \
                                                           " failed " +        \
                                         (MSG) + "\n" + st.ToString());        \
    }                                                                          \
  } while (false)

#define COMMON_RETURN_IF_FAILED_CHOOSER(x, A, B, FUNC, ...) FUNC

/**
 * @brief COMMON_RETURN_IF_FAILED(EXPR) or COMMON_RETURN_IF_FAILED(EXPR,
 * MESSAGE)
 *
 */
#define COMMON_RETURN_IF_FAILED(...)                                           \
  COMMON_RETURN_IF_FAILED_CHOOSER(, ##__VA_ARGS__,                             \
                                  COMMON_RETURN_IF_FAILED_2(__VA_ARGS__),      \
                                  COMMON_RETURN_IF_FAILED_1(__VA_ARGS__))

#define COMMON_CONSTRUCT_OR_RETURN_IMPL_1(VAR, EXPR)                           \
  auto VAR##_status = (EXPR);                                                  \
  if (!VAR##_status.ok()) {                                                    \
    return absl::Status(VAR##_status.status().code(),                          \
                        COMMON_DEBUG_LOCATION " " #EXPR " failed\n" +          \
                            VAR##_status.status().ToString());                 \
  }                                                                            \
  auto VAR = std::move(VAR##_status.value())

#define COMMON_CONSTRUCT_OR_RETURN_IMPL_2(VAR, EXPR, MSG)                      \
  auto VAR##_status = (EXPR);                                                  \
  if (!VAR##_status.ok()) {                                                    \
    return absl::Status(VAR##_status.status().code(),                          \
                        COMMON_DEBUG_LOCATION " " #EXPR " failed, err_msg: " + \
                            MSG + " \n" + VAR##_status.status().ToString());   \
  }                                                                            \
  auto VAR = std::move(VAR##_status.value())

#define COMMON_CONSTRUCT_OR_RETURN_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define COMMON_CONSTRUCT_OR_RETURN(...)                                        \
  COMMON_CONSTRUCT_OR_RETURN_GET_MACRO(__VA_ARGS__,                            \
                                       COMMON_CONSTRUCT_OR_RETURN_IMPL_2,      \
                                       COMMON_CONSTRUCT_OR_RETURN_IMPL_1)      \
  (__VA_ARGS__)

#define COMMON_COMBINE(X, Y) COMMON_COMBINE_HELPER(X, Y)
#define COMMON_COMBINE_HELPER(X, Y) X##Y

#define COMMON_ASSIGN_OR_RETURN_INTERNAL(TMP_VAR, EXPR, VAR)                   \
  COMMON_CONSTRUCT_OR_RETURN(TMP_VAR, EXPR);                                   \
  VAR = std::move(TMP_VAR)

// 检查 EXPR 的执行结果，如果 status 的状态是 ok，将结果保存到 VAR 中
//
// * EXPR 的返回值必须是 absl::StatusOr<T> 类型
#define COMMON_ASSIGN_OR_RETURN(VAR, EXPR)                                     \
  COMMON_ASSIGN_OR_RETURN_INTERNAL(COMMON_COMBINE(tmp_var_, __LINE__), EXPR,   \
                                   VAR)

#define COMMON_ASSIGN_OR_RETURN_IMPL_1(VAR, EXPR)                              \
  auto COMMON_COMBINE(assign_or_return_var, __LINE__) = (EXPR);                \
  if (!COMMON_COMBINE(assign_or_return_var, __LINE__).ok()) {                  \
    throw std::runtime_error(                                                  \
        COMMON_DEBUG_LOCATION " " #EXPR " failed\n" +                          \
        COMMON_COMBINE(assign_or_return_var, __LINE__).status().ToString());   \
  }                                                                            \
  VAR = std::move(COMMON_COMBINE(assign_or_return_var, __LINE__).value())

#define COMMON_ASSIGN_OR_RETURN_IMPL_2(VAR, EXPR, MSG)                         \
  auto VAR##_status = (EXPR);                                                  \
  if (!VAR##_status.ok()) {                                                    \
    throw std::runtime_error(                                                  \
        COMMON_DEBUG_LOCATION " " #EXPR " failed, err_msg: " + MSG + " \n" +   \
        VAR##_status.status().ToString());                                     \
  }                                                                            \
  VAR = std::move(VAR##_status.value())

#define COMMON_ASSIGN_OR_THROW_GET_MACRO(_1, _2, _3, NAME, ...) NAME

#define COMMON_ASSIGN_OR_THROW(...)                                            \
  COMMON_ASSIGN_OR_THROW_GET_MACRO(__VA_ARGS__,                                \
                                   COMMON_ASSIGN_OR_RETURN_IMPL_2,             \
                                   COMMON_ASSIGN_OR_RETURN_IMPL_1)             \
  (__VA_ARGS__)

#define COMMON_SCOPED_GUARD(EXPR)                                              \
  auto COMMON_COMBINE(scoped_guard_, __LINE__) =                               \
      ::waymax_cpp::on_scope_exited((EXPR))

#define COMMON_ON_SCOPE_EXIT(EXPR) COMMON_SCOPED_GUARD(EXPR)

#define COMMON_THROW_IF_FAILED(EXPR)                                           \
  do {                                                                         \
    auto st = (EXPR);                                                          \
    if (!st.ok()) {                                                            \
      throw std::runtime_error(st.ToString());                                 \
    }                                                                          \
  } while (false)

//
#define COMMON_CONSTRUCT_OR_THROW_IMPL_1(VAR, EXPR)                            \
  auto VAR##_status = (EXPR);                                                  \
  if (!VAR##_status.ok()) {                                                    \
    throw std::runtime_error(VAR##_status.status().ToString() +                \
                             ", expr: " + #EXPR);                              \
  }                                                                            \
  auto VAR = std::move(VAR##_status.value())

#define COMMON_CONSTRUCT_OR_THROW_IMPL_2(VAR, EXPR, ERROR_MESSAGE)             \
  auto VAR##_status = (EXPR);                                                  \
  if (!VAR##_status.ok()) {                                                    \
    throw std::runtime_error(VAR##_status.status().ToString() +                \
                             ", expr: " + #EXPR + ERROR_MESSAGE);              \
  }                                                                            \
  auto VAR = std::move(VAR##_status.value())

#define COMMON_CONSTRUCT_OR_THROW_GET_MACRO(_1, _2, _3, NAME, ...) NAME

#define COMMON_CONSTRUCT_OR_THROW(...)                                         \
  COMMON_CONSTRUCT_OR_RETURN_GET_MACRO(__VA_ARGS__,                            \
                                       COMMON_CONSTRUCT_OR_THROW_IMPL_2,       \
                                       COMMON_CONSTRUCT_OR_THROW_IMPL_1)       \
  (__VA_ARGS__)