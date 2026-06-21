# Future Work

## 1. Recheck and Validate Calculation Methods

- Review the formulas for speed score, movement quality, tremor power, LNU risk, dominant-hand prediction, and motor age.
- Confirm that all thresholds, constants, and weighting values are appropriate for upper limb dexterity screening.
- Resolve documentation differences, such as different default values for speed scoring and tremor analysis.
- Check whether the current formulas are understandable and reproducible from the exported CSV data.

## 2. Run More Sample Trials

- Collect more trial data from participants across different age groups, body sizes, and motor ability levels.
- Run repeated trials with the same participants to evaluate test-retest reliability.
- Compare results across different test durations, such as 30 seconds, 1 minute, and 3 minutes.
- Identify whether the current number of hits per session is enough for reliable statistical analysis.

## 3. Consult Clinical and Rehabilitation Experts

- Ask occupational therapists, physiotherapists, geriatric specialists, or neurologists to review the assessment workflow.
- Confirm whether LNU risk, tremor power, motor age, and movement quality are clinically meaningful as screening indicators.
- Review whether the task design is suitable for older adults and people with limited arm mobility.
- Refine wording in reports and UI so the tool remains clearly positioned as a screening aid, not a diagnostic device.

## 4. Improve Calibration

- Add a pre-test calibration step for participant distance, body position, and camera angle.
- Adjust the 3x3 grid based on the participant's body position instead of always using the full camera frame.
- Detect whether both hands, shoulders, and upper body are visible before starting the assessment.
- Provide feedback when lighting, camera placement, or participant position may reduce tracking quality.

## 5. Improve Measurement Accuracy

- Validate MediaPipe hand tracking accuracy during fast reaching movements.
- Reduce false hits caused by tracking noise or brief accidental entry into a target zone.
- Review the dwell-frame requirement and target timeout duration using real trial data.
- Improve tremor and jerk calculations for short reaches where the number of trajectory points is limited.

## 6. Expand Cross-Platform Support

- Add fallback behavior for non-Windows operating systems.
- Replace or guard Windows-specific features such as `winsound` and always-on-top OpenCV window handling.
- Test the application on macOS and Linux with different webcam devices.
- Make camera device selection configurable instead of assuming webcam index `0`.

## 7. Improve Reporting and Interpretation

- Add clearer explanations of each metric in the PDF report.
- Include reliability warnings when the number of successful hits is too small.
- Add session-to-session comparison for participants who complete multiple assessments.
- Improve chart labels and summary text so non-technical users can understand the result.

## 8. Validate with Benchmark Data

- Compare system results against manual observation or expert-scored assessments.
- Benchmark selected metrics against established upper limb or rehabilitation assessment methods where appropriate.
- Build a larger anonymized dataset to estimate normal ranges and expected variability.
- Evaluate whether the dominant-hand prediction matches participant self-report.

## 9. Improve Usability

- Add clearer setup instructions before the assessment starts.
- Provide real-time warnings when hands are not detected reliably.
- Improve error messages for camera access, missing model files, and failed report export.
- Make the restart and repeated-trial workflow easier for demonstrations or research sessions.

## 10. Strengthen Privacy and Data Handling

- Define how participant names, ages, CSV files, and PDF reports should be stored.
- Add anonymized participant IDs for research trials.
- Document privacy considerations for webcam-based assessment data.
- Consider options for local-only data storage and controlled export.
