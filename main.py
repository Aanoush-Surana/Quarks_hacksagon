def inference_thread(self):
    logger.info("Starting Optimized Temporal-Fusion Inference Thread.")

    DETECTION_INTERVAL = 2

    last_frame_data = None
    last_seg_frame = None

    while self.running:
        try:
            item = self.capture_queue.get(timeout=0.5)
            frame_idx, timestamp, frame = item
        except Empty:
            if self.capture_done:
                break
            continue

        proc_frame = self.preprocessor.process_frame(frame)
        H, W = proc_frame.shape[:2]

        t0 = time.perf_counter()

        # ---------------------------------------
        # 1. DETECTION DECIMATION (SINGLE INFERENCE)
        # ---------------------------------------
        if frame_idx % DETECTION_INTERVAL == 0 or last_frame_data is None:
            seg_frame, frame_data = self.seg_model.process_frame(
                proc_frame, frame_idx, timestamp
            )

            last_frame_data = frame_data
            last_seg_frame = seg_frame

        else:
            import copy
            frame_data = copy.deepcopy(last_frame_data)
            seg_frame = proc_frame  # fresh frame (no stale overlays)

            frame_data["frame_idx"] = frame_idx
            frame_data["timestamp"] = timestamp

        # ---------------------------------------
        # 2. EXTRACT DETECTIONS
        # ---------------------------------------
        raw_dets = frame_data.get("detections", [])

        # ---------------------------------------
        # 3. PREFILTER
        # ---------------------------------------
        fusion_states = self.fusion.get_states()

        clean_dets, suppressed_dets, stuff_dets = self.prefilter.filter(
            raw_dets, frame_idx, fusion_states
        )

        # ---------------------------------------
        # 4. SEGMENTATION SKIP (fusion logic)
        # ---------------------------------------
        skip_ids = self.fusion.get_seg_skip_set(frame_idx)

        # ---------------------------------------
        # 5. TEMPORAL FUSION
        # ---------------------------------------
        outputs = self.fusion.update(
            clean_dets,
            suppressed_dets,
            (H, W),
            frame_idx,
            skip_ids=skip_ids,
            stuff_detections=stuff_dets
        )

        # ---------------------------------------
        # 6. CLASS STABILIZATION
        # ---------------------------------------
        for det in clean_dets:
            tid = det.get("track_id")
            if tid is not None:
                s_cid, s_cname = self.stabilizer.stabilize(
                    tid,
                    det["class_id"],
                    det["class_name"],
                    det["confidence"]
                )

                if tid in outputs:
                    outputs[tid]["stable_class_name"] = s_cname
                    outputs[tid]["stable_class_id"] = s_cid

        # ---------------------------------------
        # 7. MASK POST-PROCESS
        # ---------------------------------------
        outputs = project_and_fill(outputs, (H, W))

        # ---------------------------------------
        # 8. FINAL RENDER
        # ---------------------------------------
        seg_frame, frame_data = self.seg_model.render_fusion_outputs(
            proc_frame, outputs, frame_idx, timestamp
        )

        # ---------------------------------------
        # 9. TRACKING STATS (lightweight)
        # ---------------------------------------
        tracked_frame = self.tracker.process_frame(seg_frame, frame_data)

        # ---------------------------------------
        # 10. METRICS
        # ---------------------------------------
        inference_ms = (time.perf_counter() - t0) * 1000

        if "tracking_stats" in frame_data:
            frame_data["tracking_stats"]["inference_ms"] = round(inference_ms, 1)

        # ---------------------------------------
        # 11. PERIODIC LOGGING
        # ---------------------------------------
        if frame_idx > 0 and frame_idx % 50 == 0:
            logger.info(
                f"[Frame {frame_idx}] "
                f"Fusion: {self.fusion.get_metrics()} | "
                f"Stability: {self.stabilizer.get_stability_report()} | "
                f"Flicker: {self.prefilter.get_flicker_stats()}"
            )

        # ---------------------------------------
        # 12. CLEANUP
        # ---------------------------------------
        if frame_idx > 0 and frame_idx % 100 == 0:
            active_ids = {
                det.get("track_id")
                for det in clean_dets
                if det.get("track_id") is not None
            }

            self.fusion.cleanup(active_ids, frame_idx)

            for tid in list(self.stabilizer._tracks.keys()):
                if tid not in active_ids and tid not in self.fusion._state:
                    self.stabilizer.reset(tid)

        # ---------------------------------------
        # OUTPUT PIPELINE
        # ---------------------------------------
        self.inference_queue.put((frame_idx, tracked_frame))
        self.json_queue.put(frame_data)

        if self.display_queue.empty():
            self.display_queue.put((frame_idx, tracked_frame, frame_data))

        self.capture_queue.task_done()
        self.frames_processed += 1

    self.inference_done = True
    logger.info("Inference thread finished.")