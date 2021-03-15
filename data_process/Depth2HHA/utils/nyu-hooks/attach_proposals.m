function rec = attach_proposals(voc_rec, boxes, class_to_id, num_classes)
  % ------------------------------------------------------------------------
  %           gt: [2108x1 double]
  %      overlap: [2108x20 single]
  %      dataset: 'voc_2007_trainval'
  %        boxes: [2108x4 single]
  %         feat: [2108x9216 single]
  %        class: [2108x1 uint8]
  if isfield(voc_rec, 'objects')
    gt_boxes = cat(1, voc_rec.objects(:).bbox);
    all_boxes = cat(1, gt_boxes, boxes);
    ind = isKey(class_to_id, {voc_rec.objects(:).class});
    gtc = class_to_id.values({voc_rec.objects(ind).class});
    gtc = cat(1, gtc{:});
    gt_classes = zeros(length(voc_rec.objects), 1);
    gt_classes(ind) = gtc;
    num_gt_boxes = size(gt_boxes, 1);
  else
    gt_boxes = [];
    all_boxes = boxes;
    gt_classes = [];
    num_gt_boxes = 0;
  end
  num_boxes = size(boxes, 1);

  rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
  rec.overlap = zeros(num_gt_boxes+num_boxes, num_classes, 'single');
  for i = 1:num_gt_boxes
    if(gt_classes(i) > 0)
      rec.overlap(:, gt_classes(i)) = ...
          max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
    end
  end
  rec.boxes = single(all_boxes);
  rec.feat = [];
  rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
end
