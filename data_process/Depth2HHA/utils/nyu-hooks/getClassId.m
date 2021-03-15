function classId = getClassId(className, classMapping)
  dt = getMetadata(classMapping);
  for i = 1:length(className),
    classId(i) = find(strcmp(dt.className, className{i}));
  end
end
