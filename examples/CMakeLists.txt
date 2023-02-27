#
# 为每个Cpp文件单独创建一个exe
# 
file(GLOB_RECURSE files "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
foreach(file ${files})  #遍历
  #获得文件名
  get_filename_component(name ${file} NAME_WE)
  #获得目录
  STRING(REGEX REPLACE "${CMAKE_CURRENT_LIST_DIR}(.+)/.+" "\\1" DIR ${file})

  #创建exe 
	add_executable(${name} ${file})
  #链接CGAL
  target_link_libraries(${name} PRIVATE OpenCV)
  #指定此Target的目录，在VS中才能分组
  set_property(TARGET ${name} PROPERTY FOLDER ${DIR})

  #数据路径
  target_compile_definitions(${name} PRIVATE "-DDATA_PATH=\"${DataDir}\"")
endforeach()