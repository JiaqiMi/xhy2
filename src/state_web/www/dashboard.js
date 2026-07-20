"use strict";

const dashboardState = {
    status: null,
    connected: false,
    mapScale: 20,
    mapPanX: 0,
    mapPanY: 0,
    zScale: 20,
    zPanY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragPanX: 0,
    dragPanY: 0,
    zDragging: false,
    zDragStartY: 0,
    zDragPanY: 0,
};


function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}


function numberText(value, digits = 2, suffix = "") {
    const number = finiteNumber(value);
    return number === null ? "--" : `${number.toFixed(digits)}${suffix}`;
}


function integerText(value) {
    const number = finiteNumber(value);
    return number === null ? "--" : String(Math.round(number));
}


function ageText(value) {
    const age = finiteNumber(value);
    if (age === null) return "--";
    if (age < 1) return `${Math.round(age * 1000)} ms`;
    return `${age.toFixed(age < 10 ? 2 : 1)} s`;
}


function radToDeg(value) {
    const number = finiteNumber(value);
    return number === null ? null : number * 180 / Math.PI;
}


function numericDifference(target, actual) {
    const targetNumber = finiteNumber(target);
    const actualNumber = finiteNumber(actual);
    return targetNumber === null || actualNumber === null
        ? null
        : targetNumber - actualNumber;
}


function shortestAngleDifference(target, actual) {
    const difference = numericDifference(target, actual);
    if (difference === null) return null;
    return ((difference + 540) % 360) - 180;
}


function hexadecimal(value, width = 2) {
    const number = finiteNumber(value);
    if (number === null) return "--";
    return `0x${Math.round(number).toString(16).toUpperCase().padStart(width, "0")}`;
}


function snapshotText(snapshot) {
    if (!snapshot || !snapshot.data) return "无数据";
    return snapshot.online
        ? `在线 · ${ageText(snapshot.age_sec)}`
        : `超时 · ${ageText(snapshot.age_sec)}`;
}


function snapshotClass(snapshot) {
    if (!snapshot || !snapshot.data) return "bad";
    return snapshot.online ? "good" : "stale";
}


function setRows(containerId, rows) {
    const container = document.getElementById(containerId);
    const fragment = document.createDocumentFragment();

    rows.forEach((row) => {
        const item = document.createElement("div");
        item.className = "data-item";

        const label = document.createElement("div");
        label.className = "data-label";
        label.textContent = row.label;

        const value = document.createElement("div");
        value.className = `data-value ${row.className || ""}`.trim();
        value.textContent = row.value === undefined || row.value === null
            ? "--"
            : String(row.value);
        if (row.title) value.title = row.title;

        item.append(label, value);
        fragment.appendChild(item);
    });

    container.replaceChildren(fragment);
}


function setAxisRows(containerId, rows) {
    const container = document.getElementById(containerId);
    const fragment = document.createDocumentFragment();

    rows.forEach((row) => {
        const item = document.createElement("div");
        item.className = "axis-row";

        const label = document.createElement("div");
        label.className = "axis-row-label";
        label.textContent = row.label;
        if (row.title) label.title = row.title;
        item.appendChild(label);

        row.cells.forEach((cell) => {
            const valueCell = document.createElement("div");
            valueCell.className = `axis-cell ${cell.className || ""}`.trim();
            if (cell.title) valueCell.title = cell.title;
            if (cell.span) {
                valueCell.style.gridColumn = `span ${cell.span}`;
            }

            const axis = document.createElement("span");
            axis.className = "axis-name";
            axis.textContent = cell.axis;

            const value = document.createElement("span");
            value.className = "axis-value";
            value.textContent = cell.value === undefined || cell.value === null
                ? "--"
                : String(cell.value);

            valueCell.append(axis, value);
            item.appendChild(valueCell);
        });

        fragment.appendChild(item);
    });

    container.replaceChildren(fragment);
}


function badge(label, online, alarm = false, warning = false) {
    const element = document.createElement("span");
    element.className = `badge ${
        alarm ? "alarm" : (warning ? "warning" : (online ? "online" : "offline"))
    }`;
    element.textContent = label;
    return element;
}


function renderGlobalBadges(data) {
    const container = document.getElementById("global-badges");
    const feedbackSensor = data.feedback?.data?.sensor || {};
    const power = data.power?.data || {};
    const safetyKnown = Boolean(data.feedback?.online);
    const hasAlarm = Boolean(
        feedbackSensor.leak_alarm
        || finiteNumber(feedbackSensor.fault_status) > 0
        || (data.power?.data && power.checksum_ok === false)
    );

    const fragment = document.createDocumentFragment();
    fragment.appendChild(badge("Web 在线", true));
    fragment.appendChild(badge(
        data.origin?.online ? "原点已就绪" : "等待原点",
        Boolean(data.origin?.online),
    ));
    fragment.appendChild(badge(
        data.tf?.online ? "TF 在线" : "TF 离线",
        Boolean(data.tf?.online),
    ));
    ["left", "right", "fisheye"].forEach((name) => {
        const labels = {left: "左目", right: "右目", fisheye: "鱼眼"};
        fragment.appendChild(badge(
            `${labels[name]}${data.streams?.[name]?.online ? "在线" : "离线"}`,
            Boolean(data.streams?.[name]?.online),
        ));
    });
    if (!safetyKnown) {
        fragment.appendChild(badge("安全状态未知", false, false, true));
    } else {
        fragment.appendChild(badge(
            hasAlarm ? "存在安全告警" : "安全状态正常",
            !hasAlarm,
            hasAlarm,
        ));
    }
    container.replaceChildren(fragment);
}


function renderCamera(name, stream) {
    const stateElement = document.getElementById(`camera-${name}-state`);
    const metaElement = document.getElementById(`camera-${name}-meta`);
    const imageElement = document.getElementById(`camera-${name}-image`);
    const online = Boolean(stream?.online);

    stateElement.textContent = online ? "在线" : "离线";
    stateElement.className = `camera-state ${online ? "online" : "offline"}`;
    imageElement.classList.toggle("is-online", online);

    if (!stream) {
        metaElement.textContent = "--";
        return;
    }
    const resolution = stream.width && stream.height
        ? `${stream.width}×${stream.height}`
        : "--";
    metaElement.textContent = [
        stream.topic || "--",
        resolution,
        `${numberText(stream.fps, 1)} FPS`,
        `年龄 ${ageText(stream.age_sec)}`,
    ].join(" · ");
}


function renderCoreStatus(data) {
    const tfPose = data.tf?.data || {};
    const tfPosition = tfPose.position_m || {};
    const tfOrientation = tfPose.orientation_deg || {};
    const feedback = data.feedback?.data || {};
    const actualForce = feedback.motor_force || {};
    const velocity = data.velocity?.data || {};
    const linear = velocity.linear_mps || {};
    const angular = velocity.angular_radps || {};
    const command = data.pose_command?.data || {};
    const target = command.target || {};
    const targetPosition = target.position_m || {};
    const targetOrientation = target.orientation_deg || {};
    const targetForce = command.force || {};
    const motion = data.motion_state?.data || {};
    const tfClass = snapshotClass(data.tf);
    const commandClass = snapshotClass(data.pose_command);
    const feedbackClass = snapshotClass(data.feedback);
    const velocityClass = snapshotClass(data.velocity);
    const motionClass = snapshotClass(data.motion_state);
    const poseErrorClass = (
        data.tf?.online && data.pose_command?.online
            ? ""
            : (
                data.tf?.data && data.pose_command?.data
                    ? "stale"
                    : "bad"
            )
    );

    setAxisRows("core-status", [
        {
            label: "消息状态",
            cells: [
                {
                    axis: "TF",
                    value: snapshotText(data.tf),
                    className: tfClass,
                },
                {
                    axis: "cmdned",
                    value: snapshotText(data.pose_command),
                    className: commandClass,
                },
                {
                    axis: "反馈 / 速度",
                    value: `${snapshotText(data.feedback)} / ${snapshotText(data.velocity)}`,
                    className: (
                        data.feedback?.online && data.velocity?.online
                            ? "good"
                            : (
                                data.feedback?.data || data.velocity?.data
                                    ? "stale"
                                    : "bad"
                            )
                    ),
                    title: `/status/auv：${snapshotText(data.feedback)}；`
                        + `/status/vel：${snapshotText(data.velocity)}`,
                },
            ],
        },
        {
            label: "运行状态",
            cells: [
                {
                    axis: "debug_driver",
                    value: feedback.control_mode_name
                        ? `${feedback.control_mode_name} (${feedback.control_mode})`
                        : "--",
                    className: feedbackClass,
                },
                {
                    axis: "motion_state",
                    value: motion.state_name
                        ? `${motion.state_name} (${motion.state})`
                        : "--",
                    className: motionClass,
                },
                {
                    axis: "状态原因",
                    value: motion.reason || "--",
                    className: motionClass,
                    title: motion.reason || "",
                },
            ],
        },
        {
            label: "实际位置",
            cells: [
                {axis: "X / North", value: numberText(tfPosition.x, 3, " m"), className: tfClass},
                {axis: "Y / East", value: numberText(tfPosition.y, 3, " m"), className: tfClass},
                {axis: "Z / Down", value: numberText(tfPosition.z, 3, " m"), className: tfClass},
            ],
        },
        {
            label: "目标位置",
            cells: [
                {axis: "X / North", value: numberText(targetPosition.x, 3, " m"), className: commandClass},
                {axis: "Y / East", value: numberText(targetPosition.y, 3, " m"), className: commandClass},
                {axis: "Z / Down", value: numberText(targetPosition.z, 3, " m"), className: commandClass},
            ],
        },
        {
            label: "位置误差",
            title: "目标位置减实际 TF 位置",
            cells: [
                {axis: "ΔX", value: numberText(numericDifference(targetPosition.x, tfPosition.x), 3, " m"), className: poseErrorClass},
                {axis: "ΔY", value: numberText(numericDifference(targetPosition.y, tfPosition.y), 3, " m"), className: poseErrorClass},
                {axis: "ΔZ", value: numberText(numericDifference(targetPosition.z, tfPosition.z), 3, " m"), className: poseErrorClass},
            ],
        },
        {
            label: "实际姿态",
            cells: [
                {axis: "Roll", value: numberText(tfOrientation.roll_deg, 2, "°"), className: tfClass},
                {axis: "Pitch", value: numberText(tfOrientation.pitch_deg, 2, "°"), className: tfClass},
                {axis: "Heading", value: numberText(tfOrientation.heading_deg, 2, "°"), className: tfClass},
            ],
        },
        {
            label: "目标姿态",
            cells: [
                {axis: "Roll", value: numberText(targetOrientation.roll_deg, 2, "°"), className: commandClass},
                {axis: "Pitch", value: numberText(targetOrientation.pitch_deg, 2, "°"), className: commandClass},
                {axis: "Heading", value: numberText(targetOrientation.heading_deg, 2, "°"), className: commandClass},
            ],
        },
        {
            label: "姿态误差",
            title: "目标 Yaw 减实际 TF Yaw，范围为 [-180°, 180°)",
            cells: [
                {
                    axis: "ΔYaw（目标 − 实际）",
                    value: numberText(
                        shortestAngleDifference(
                            targetOrientation.heading_deg,
                            tfOrientation.heading_deg,
                        ),
                        2,
                        "°",
                    ),
                    className: poseErrorClass,
                    span: 3,
                },
            ],
        },
        {
            label: "实际力 / 力矩",
            title: "每列依次显示平移力 T 与旋转力矩 M",
            cells: [
                {axis: "TX / MX", value: `${integerText(actualForce.tx)} / ${integerText(actualForce.mx)}`, className: feedbackClass},
                {axis: "TY / MY", value: `${integerText(actualForce.ty)} / ${integerText(actualForce.my)}`, className: feedbackClass},
                {axis: "TZ / MZ", value: `${integerText(actualForce.tz)} / ${integerText(actualForce.mz)}`, className: feedbackClass},
            ],
        },
        {
            label: "目标力 / 力矩",
            title: "cmdned 指令；每列依次显示平移力 T 与旋转力矩 M",
            cells: [
                {axis: "TX / MX", value: `${integerText(targetForce.tx)} / ${integerText(targetForce.mx)}`, className: commandClass},
                {axis: "TY / MY", value: `${integerText(targetForce.ty)} / ${integerText(targetForce.my)}`, className: commandClass},
                {axis: "TZ / MZ", value: `${integerText(targetForce.tz)} / ${integerText(targetForce.mz)}`, className: commandClass},
            ],
        },
        {
            label: "线速度",
            cells: [
                {axis: "X", value: numberText(linear.x, 3, " m/s"), className: velocityClass},
                {axis: "Y", value: numberText(linear.y, 3, " m/s"), className: velocityClass},
                {axis: "Z", value: numberText(linear.z, 3, " m/s"), className: velocityClass},
            ],
        },
        {
            label: "角速度",
            cells: [
                {axis: "X", value: numberText(radToDeg(angular.x), 2, "°/s"), className: velocityClass},
                {axis: "Y", value: numberText(radToDeg(angular.y), 2, "°/s"), className: velocityClass},
                {axis: "Z", value: numberText(radToDeg(angular.z), 2, "°/s"), className: velocityClass},
            ],
        },
    ]);
}


function renderMotionState(data) {
    const motion = data.motion_state?.data || {};
    const force = motion.force || {};

    setRows("motion-status", [
        {
            label: "状态话题",
            value: snapshotText(data.motion_state),
            className: snapshotClass(data.motion_state),
        },
        {
            label: "当前状态",
            value: motion.state_name
                ? `${motion.state_name} (${motion.state})`
                : "--",
        },
        {
            label: "目标有效",
            value: motion.goal_active === undefined
                ? "--"
                : (motion.goal_active ? "是" : "否"),
        },
        {label: "位置误差", value: numberText(motion.position_error_m, 3, " m")},
        {label: "航向误差", value: numberText(radToDeg(motion.yaw_error_rad), 2, "°")},
        {label: "水平速度", value: numberText(motion.horizontal_speed_mps, 3, " m/s")},
        {label: "航向角速度", value: numberText(radToDeg(motion.yaw_rate_radps), 2, "°/s")},
        {
            label: "监督输出",
            value: `TX ${integerText(force.tx)} · TY ${integerText(force.ty)} · MZ ${integerText(force.mz)}`,
        },
        {
            label: "状态原因",
            value: motion.reason || "--",
            title: motion.reason || "",
        },
    ]);
}


function renderActuatorStatus(data) {
    const command = data.actuator_command?.data || {};
    const feedback = data.actuator_feedback?.data || {};

    setRows("actuator-status", [
        {
            label: "指令状态",
            value: snapshotText(data.actuator_command),
            className: snapshotClass(data.actuator_command),
        },
        {
            label: "反馈状态",
            value: snapshotText(data.actuator_feedback),
            className: snapshotClass(data.actuator_feedback),
        },
        {label: "指令模式", value: command.mode_name || "--"},
        {label: "补光灯1 指令/状态", value: `${integerText(command.light1)} / ${integerText(feedback.light1)}`},
        {label: "补光灯2 指令/状态", value: `${integerText(command.light2)} / ${integerText(feedback.light2)}`},
        {label: "航向舵机 指令/反馈", value: `${integerText(command.heading_servo)} / ${integerText(feedback.heading_servo)}`},
        {label: "夹爪舵机 指令/反馈", value: `${integerText(command.clamp_servo)} / ${integerText(feedback.clamp_servo)}`},
        {label: "推杆动作 指令/反馈", value: `${integerText(command.drive_cmd)} / ${integerText(feedback.drive_cmd)}`},
        {label: "推杆速度 指令/反馈", value: `${integerText(command.drive_speed)} / ${integerText(feedback.drive_speed)}`},
        {label: "红灯 指令/反馈", value: `${integerText(command.red_light)} / ${integerText(feedback.red_light)}`},
        {label: "黄灯 指令/反馈", value: `${integerText(command.yellow_light)} / ${integerText(feedback.yellow_light)}`},
        {label: "绿灯 指令/反馈", value: `${integerText(command.green_light)} / ${integerText(feedback.green_light)}`},
    ]);
}


function renderMotionDiagnostics(data) {
    const diagnostics = data.motion_diagnostics?.data || {};
    const vectorSpeed = Math.hypot(
        finiteNumber(diagnostics.reference_velocity_x) || 0,
        finiteNumber(diagnostics.reference_velocity_y) || 0,
    );
    setRows("motion-diagnostics", [
        {
            label: "诊断话题",
            value: snapshotText(data.motion_diagnostics),
            className: snapshotClass(data.motion_diagnostics),
        },
        {
            label: "地图系速度",
            value: `N ${numberText(diagnostics.map_velocity_x, 3)} / E ${numberText(diagnostics.map_velocity_y, 3)} m/s`,
        },
        {
            label: "XY 速度参考",
            value: `N ${numberText(diagnostics.reference_velocity_x, 3)} / E ${numberText(diagnostics.reference_velocity_y, 3)} m/s，|v| ${numberText(vectorSpeed, 3)} m/s`,
        },
        {
            label: "闭合速度 / 停止距离",
            value: `${numberText(diagnostics.closing_speed, 3)} m/s / ${numberText(diagnostics.xy_stop_distance, 3)} m`,
            className: diagnostics.xy_braking ? "warning" : "",
        },
        {
            label: "XY 主动制动",
            value: diagnostics.xy_braking === undefined ? "--" : (diagnostics.xy_braking ? "进行中" : "跟踪中"),
            className: diagnostics.xy_braking ? "warning" : "good",
        },
        {
            label: "Yaw 速度参考 / 停止角",
            value: `${numberText(radToDeg(diagnostics.yaw_rate_reference), 2)} °/s / ${numberText(radToDeg(diagnostics.yaw_stop_angle), 2)} °`,
        },
        {
            label: "Yaw 主动制动",
            value: diagnostics.yaw_braking === undefined ? "--" : (diagnostics.yaw_braking ? "进行中" : "跟踪中"),
            className: diagnostics.yaw_braking ? "warning" : "good",
        },
        {
            label: "目标几何静止",
            value: `${numberText(diagnostics.goal_static_seconds, 2)} s${diagnostics.goal_static_for_capture ? "，允许接管" : "，禁止接管"}`,
        },
        {
            label: "限幅前指令",
            value: `TX ${numberText(diagnostics.raw_tx, 0)} / TY ${numberText(diagnostics.raw_ty, 0)} / MZ ${numberText(diagnostics.raw_mz, 0)}`,
        },
    ]);
}


function renderPowerStatus(data) {
    const power = data.power?.data || {};
    const power1 = power.power1 || {};
    const power2 = power.power2 || {};
    const sensor = data.feedback?.data?.sensor || {};
    const leak = sensor.leak_alarm;
    const fault = finiteNumber(sensor.fault_status);

    setRows("power-status", [
        {
            label: "电源话题",
            value: snapshotText(data.power),
            className: snapshotClass(data.power),
        },
        {
            label: "校验",
            value: power.checksum_ok === undefined
                ? "--"
                : (power.checksum_ok ? "正常" : "失败"),
            className: power.checksum_ok === undefined
                ? ""
                : (power.checksum_ok ? "good" : "bad"),
        },
        {
            label: "电源1 有效",
            value: power1.valid === undefined ? "--" : (power1.valid ? "是" : "否"),
            className: power1.valid === false ? "bad" : "",
        },
        {label: "电源1 电压", value: numberText(power1.voltage_v, 2, " V")},
        {label: "电源1 电流", value: numberText(power1.current_a, 2, " A")},
        {label: "电源1 功率", value: numberText(power1.power_w, 2, " W")},
        {
            label: "电源2 有效",
            value: power2.valid === undefined ? "--" : (power2.valid ? "是" : "否"),
            className: power2.valid === false ? "bad" : "",
        },
        {label: "电源2 电压", value: numberText(power2.voltage_v, 2, " V")},
        {label: "电源2 电流", value: numberText(power2.current_a, 2, " A")},
        {label: "电源2 功率", value: numberText(power2.power_w, 2, " W")},
        {label: "舱内温度", value: numberText(sensor.temperature_c, 1, " ℃")},
        {label: "控制电压", value: numberText(sensor.voltage_v, 2, " V")},
        {label: "系统电流", value: numberText(sensor.current_a, 2, " A")},
        {
            label: "漏水告警",
            value: leak === undefined ? "--" : (leak ? "告警" : "正常"),
            className: leak === undefined ? "" : (leak ? "bad" : "good"),
        },
        {
            label: "故障状态",
            value: hexadecimal(sensor.fault_status, 4),
            className: fault === null ? "" : (fault > 0 ? "bad" : "good"),
        },
        {label: "传感器有效位", value: hexadecimal(sensor.sensor_valid, 2)},
        {label: "传感器更新位", value: hexadecimal(sensor.sensor_updated, 2)},
        {label: "设备电源位", value: hexadecimal(sensor.power_status, 4)},
    ]);
}


function renderSystemStatus(data) {
    const origin = data.origin?.data || {};
    const rows = [
        {
            label: "坐标系状态",
            value: data.ready ? "已就绪" : "等待原点或 TF",
            className: data.ready ? "good" : "bad",
        },
        {label: "世界坐标系", value: data.frames?.world || "--"},
        {label: "机器人坐标系", value: data.frames?.base || "--"},
        {label: "原点版本", value: integerText(origin.revision)},
        {label: "原点纬度", value: numberText(origin.latitude_deg, 7, "°")},
        {label: "原点经度", value: numberText(origin.longitude_deg, 7, "°")},
        {label: "原点深度", value: numberText(origin.depth_m, 3, " m")},
    ];

    Object.entries(data.topic_health || {}).forEach(([name, health]) => {
        rows.push({
            label: name,
            value: `${health.online ? "在线" : "离线"} · ${ageText(health.age_sec)}`,
            className: health.online ? "good" : "bad",
            title: health.topic || "",
        });
    });
    setRows("system-status", rows);
}


function resizeCanvas(canvas) {
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    const deviceWidth = Math.max(1, Math.floor(width * ratio));
    const deviceHeight = Math.max(1, Math.floor(height * ratio));

    if (canvas.width !== deviceWidth || canvas.height !== deviceHeight) {
        canvas.width = deviceWidth;
        canvas.height = deviceHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {context, width, height};
}


function niceDistance(rawDistance) {
    if (!Number.isFinite(rawDistance) || rawDistance <= 0) return 1;
    const exponent = Math.floor(Math.log10(rawDistance));
    const base = 10 ** exponent;
    const fraction = rawDistance / base;
    let niceFraction = 1;
    if (fraction > 5) niceFraction = 10;
    else if (fraction > 2) niceFraction = 5;
    else if (fraction > 1) niceFraction = 2;
    return niceFraction * base;
}


function snapshotAnnotation(snapshot) {
    if (!snapshot || !snapshot.data) return "无数据";
    return snapshot.online ? "" : `已超时 ${ageText(snapshot.age_sec)}`;
}


function drawDirectionalPose(ctx, screen, heading, options) {
    const {
        color,
        label,
        marker = "circle",
        labelOffsetY = 8,
    } = options;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.4;

    if (marker === "diamond") {
        ctx.beginPath();
        ctx.moveTo(screen.x, screen.y - 7);
        ctx.lineTo(screen.x + 7, screen.y);
        ctx.lineTo(screen.x, screen.y + 7);
        ctx.lineTo(screen.x - 7, screen.y);
        ctx.closePath();
        ctx.globalAlpha = 0.34;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.stroke();
    } else {
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#031018";
        ctx.stroke();
        ctx.strokeStyle = color;
    }

    if (heading !== null) {
        const radians = heading * Math.PI / 180;
        const arrowLength = 28;
        const tipX = screen.x + Math.sin(radians) * arrowLength;
        const tipY = screen.y - Math.cos(radians) * arrowLength;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(screen.x, screen.y);
        ctx.lineTo(tipX, tipY);
        ctx.stroke();

        const headAngle = Math.PI / 7;
        const headLength = 7;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(
            tipX - Math.sin(radians - headAngle) * headLength,
            tipY + Math.cos(radians - headAngle) * headLength,
        );
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(
            tipX - Math.sin(radians + headAngle) * headLength,
            tipY + Math.cos(radians + headAngle) * headLength,
        );
        ctx.stroke();
    }

    ctx.fillStyle = color;
    ctx.font = "bold 11px Microsoft YaHei, Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(label, screen.x + 10, screen.y + labelOffsetY);
    ctx.restore();
}


function drawActualFrameArrow(ctx, points, heading, options) {
    const {
        color,
        label,
        frameNames,
    } = options;
    const {base, camera} = points;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;

    // 使用真实 TF 点位从机体中心连接到相机。
    ctx.beginPath();
    ctx.moveTo(base.x, base.y);
    ctx.lineTo(camera.x, camera.y);
    ctx.stroke();

    let directionX = camera.x - base.x;
    let directionY = camera.y - base.y;
    let directionLength = Math.hypot(directionX, directionY);
    if (directionLength < 0.5 && heading !== null) {
        const radians = heading * Math.PI / 180;
        directionX = Math.sin(radians);
        directionY = -Math.cos(radians);
        directionLength = 1;
    }

    // 箭头尖端严格落在 camera 坐标点。
    if (directionLength >= 0.5) {
        const unitX = directionX / directionLength;
        const unitY = directionY / directionLength;
        const normalX = -unitY;
        const normalY = unitX;
        const headLength = 10;
        const headWidth = 5;
        ctx.beginPath();
        ctx.moveTo(camera.x, camera.y);
        ctx.lineTo(
            camera.x - unitX * headLength + normalX * headWidth,
            camera.y - unitY * headLength + normalY * headWidth,
        );
        ctx.moveTo(camera.x, camera.y);
        ctx.lineTo(
            camera.x - unitX * headLength - normalX * headWidth,
            camera.y - unitY * headLength - normalY * headWidth,
        );
        ctx.stroke();
    }

    // 起点和箭头端分别标出 base_link 与 camera 坐标系。
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(base.x, base.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#031018";
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = "bold 10px Microsoft YaHei, Consolas, monospace";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(3, 16, 24, 0.94)";

    ctx.textBaseline = "bottom";
    ctx.textAlign = "center";
    ctx.strokeText(frameNames.base, base.x, base.y - 10);
    ctx.fillText(frameNames.base, base.x, base.y - 10);

    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.strokeText(frameNames.camera, camera.x + 8, camera.y + 7);
    ctx.fillText(frameNames.camera, camera.x + 8, camera.y + 7);

    ctx.font = "bold 11px Microsoft YaHei, Consolas, monospace";
    ctx.strokeText(label, base.x + 10, base.y + 25);
    ctx.fillText(label, base.x + 10, base.y + 25);
    ctx.restore();
}


function drawXYMap(data) {
    const canvas = document.getElementById("xy-canvas");
    const {context: ctx, width, height} = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    const scale = dashboardState.mapScale;
    const originX = width / 2 + dashboardState.mapPanX;
    const originY = height / 2 + dashboardState.mapPanY;
    const gridStep = niceDistance(70 / scale);
    const eastMin = -originX / scale;
    const eastMax = (width - originX) / scale;
    const northMin = (originY - height) / scale;
    const northMax = originY / scale;

    ctx.lineWidth = 1;
    ctx.font = "10px Consolas, monospace";
    ctx.textBaseline = "top";

    for (
        let east = Math.ceil(eastMin / gridStep) * gridStep;
        east <= eastMax + gridStep * 0.5;
        east += gridStep
    ) {
        const screenX = originX + east * scale;
        const isAxis = Math.abs(east) < gridStep * 0.01;
        ctx.strokeStyle = isAxis ? "#3c799c" : "#19354a";
        ctx.beginPath();
        ctx.moveTo(screenX, 0);
        ctx.lineTo(screenX, height);
        ctx.stroke();
        ctx.fillStyle = "#66849d";
        ctx.fillText(numberText(east, gridStep < 1 ? 1 : 0), screenX + 3, 3);
    }

    for (
        let north = Math.ceil(northMin / gridStep) * gridStep;
        north <= northMax + gridStep * 0.5;
        north += gridStep
    ) {
        const screenY = originY - north * scale;
        const isAxis = Math.abs(north) < gridStep * 0.01;
        ctx.strokeStyle = isAxis ? "#3c799c" : "#19354a";
        ctx.beginPath();
        ctx.moveTo(0, screenY);
        ctx.lineTo(width, screenY);
        ctx.stroke();
        ctx.fillStyle = "#66849d";
        ctx.fillText(numberText(north, gridStep < 1 ? 1 : 0), 4, screenY + 3);
    }

    const worldToScreen = (north, east) => ({
        x: originX + east * scale,
        y: originY - north * scale,
    });

    const tfData = data.tf?.data || {};
    const position = tfData.position_m;
    const north = finiteNumber(position?.x);
    const east = finiteNumber(position?.y);
    const actualScreen = north !== null && east !== null
        ? worldToScreen(north, east)
        : null;
    const actualHeading = finiteNumber(
        tfData.orientation_deg?.heading_deg,
    );
    const framePoses = tfData.frame_poses || {};
    const frameScreen = (framePose) => {
        const frameNorth = finiteNumber(framePose?.position_m?.x);
        const frameEast = finiteNumber(framePose?.position_m?.y);
        return frameNorth !== null && frameEast !== null
            ? worldToScreen(frameNorth, frameEast)
            : null;
    };
    const actualFramePoints = {
        base: frameScreen(framePoses.base) || actualScreen,
        camera: frameScreen(framePoses.camera),
    };
    const hasActualFrameArrow = Boolean(
        actualFramePoints.base
        && actualFramePoints.camera
    );

    const targetPose = data.pose_command?.data?.target;
    const targetNorth = finiteNumber(targetPose?.position_m?.x);
    const targetEast = finiteNumber(targetPose?.position_m?.y);
    const targetScreen = targetNorth !== null && targetEast !== null
        ? worldToScreen(targetNorth, targetEast)
        : null;
    const targetHeading = finiteNumber(
        targetPose?.orientation_deg?.heading_deg,
    );

    if (actualScreen && targetScreen) {
        ctx.save();
        ctx.strokeStyle = data.pose_command?.online ? "#b94ea8" : "#66727e";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([5, 5]);
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.moveTo(actualScreen.x, actualScreen.y);
        ctx.lineTo(targetScreen.x, targetScreen.y);
        ctx.stroke();
        ctx.restore();
    }

    if (targetScreen) {
        const annotation = snapshotAnnotation(data.pose_command);
        drawDirectionalPose(ctx, targetScreen, targetHeading, {
            color: data.pose_command?.online ? "#ff62cf" : "#7f8994",
            label: [
                `目标 base_link N ${numberText(targetNorth, 2)}  E ${numberText(targetEast, 2)}`,
                annotation,
            ].filter(Boolean).join(" · "),
            marker: "diamond",
            labelOffsetY: -22,
        });
    }

    if (actualScreen) {
        const annotation = snapshotAnnotation(data.tf);
        const actualColor = data.tf?.online ? "#42e7a8" : "#8a97a6";
        const actualLabel = [
            `实际 N ${numberText(north, 2)}  E ${numberText(east, 2)}`,
            annotation,
        ].filter(Boolean).join(" · ");
        if (hasActualFrameArrow) {
            drawActualFrameArrow(
                ctx,
                actualFramePoints,
                actualHeading,
                {
                    color: actualColor,
                    label: actualLabel,
                    frameNames: {
                        base: data.frames?.base || "base_link",
                        camera: data.frames?.camera || "camera",
                    },
                },
            );
        } else {
            drawDirectionalPose(ctx, actualScreen, actualHeading, {
                color: actualColor,
                label: actualLabel,
            });
        }
    }

    const scaleDistance = niceDistance(110 / scale);
    const scalePixels = scaleDistance * scale;
    const barX = 16;
    const barY = height - 20;
    ctx.strokeStyle = "#f1f7fc";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(barX, barY);
    ctx.lineTo(barX + scalePixels, barY);
    ctx.moveTo(barX, barY - 5);
    ctx.lineTo(barX, barY + 5);
    ctx.moveTo(barX + scalePixels, barY - 5);
    ctx.lineTo(barX + scalePixels, barY + 5);
    ctx.stroke();
    ctx.fillStyle = "#f1f7fc";
    ctx.fillText(
        `${numberText(scaleDistance, scaleDistance < 1 ? 1 : 0)} m`,
        barX,
        barY - 18,
    );

    ctx.fillStyle = "#8fb4ce";
    ctx.fillText("Y / East →", Math.max(8, width - 80), height - 18);
    ctx.save();
    ctx.translate(12, 72);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("X / North →", 0, 0);
    ctx.restore();

    const notices = [];
    if (!data.tf?.online) {
        notices.push(`实际位姿：${snapshotAnnotation(data.tf)}`);
    }
    if (!data.pose_command?.online) {
        notices.push(`目标位姿：${snapshotAnnotation(data.pose_command)}`);
    }
    if (data.tf?.data && !hasActualFrameArrow) {
        const missingFrames = [
            actualFramePoints.base ? null : (data.frames?.base || "base_link"),
            actualFramePoints.camera ? null : (data.frames?.camera || "camera"),
        ].filter(Boolean);
        notices.push(`缺少 TF：${missingFrames.join("、")}`);
    }
    if (notices.length) {
        ctx.fillStyle = "rgba(7, 17, 29, 0.82)";
        ctx.fillRect(0, 0, width, 25);
        ctx.fillStyle = "#aeb9c5";
        ctx.font = "bold 11px Microsoft YaHei, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(notices.join("；"), width / 2, 7);
        ctx.textAlign = "left";
    }
}


function drawZAxis(data) {
    const canvas = document.getElementById("z-canvas");
    const {context: ctx, width, height} = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    const scale = dashboardState.zScale;
    const centerY = height / 2 + dashboardState.zPanY;
    const axisX = width * 0.48;
    const step = niceDistance(55 / scale);
    const minZ = -centerY / scale;
    const maxZ = (height - centerY) / scale;

    ctx.strokeStyle = "#3c799c";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(axisX, 12);
    ctx.lineTo(axisX, height - 12);
    ctx.stroke();

    ctx.fillStyle = "#8fb4ce";
    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "left";
    for (
        let z = Math.ceil(minZ / step) * step;
        z <= maxZ + step * 0.5;
        z += step
    ) {
        const screenY = centerY + z * scale;
        ctx.strokeStyle = Math.abs(z) < step * 0.01 ? "#7fc8f1" : "#315168";
        ctx.lineWidth = Math.abs(z) < step * 0.01 ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(axisX - 7, screenY);
        ctx.lineTo(axisX + 7, screenY);
        ctx.stroke();
        ctx.fillStyle = "#7693aa";
        ctx.fillText(
            numberText(z, step < 1 ? 1 : 0),
            axisX + 9,
            screenY - 6,
        );
    }

    const drawDepthMarker = (
        z,
        label,
        color,
        annotation,
        dashed = false,
        labelBelow = false,
    ) => {
        if (z === null) return;

        const screenY = centerY + z * scale;
        const clampedY = Math.max(12, Math.min(height - 12, screenY));
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = dashed ? 2.4 : 3;
        if (dashed) ctx.setLineDash([6, 4]);
        ctx.beginPath();
        if (dashed) {
            ctx.moveTo(axisX + 2, clampedY);
            ctx.lineTo(width - 7, clampedY);
        } else {
            ctx.moveTo(7, clampedY);
            ctx.lineTo(axisX - 2, clampedY);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        if (dashed) {
            ctx.moveTo(width - 7, clampedY);
            ctx.lineTo(width - 16, clampedY - 6);
            ctx.lineTo(width - 16, clampedY + 6);
        } else {
            ctx.moveTo(7, clampedY);
            ctx.lineTo(16, clampedY - 6);
            ctx.lineTo(16, clampedY + 6);
        }
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = color;
        ctx.font = "bold 10px Microsoft YaHei, Consolas, monospace";
        const text = `${label} ${numberText(z, 2, " m")}`;
        const labelY = labelBelow
            ? Math.min(height - 15, clampedY + 7)
            : Math.max(2, clampedY - 19);
        ctx.fillText(text, 4, labelY);
        if (annotation) {
            ctx.font = "9px Microsoft YaHei, Consolas, monospace";
            ctx.fillText(
                annotation,
                4,
                labelBelow
                    ? Math.min(height - 12, labelY + 12)
                    : Math.max(2, labelY - 11),
            );
        }
        ctx.restore();
    };

    const targetZ = finiteNumber(
        data.pose_command?.data?.target?.position_m?.z,
    );
    drawDepthMarker(
        targetZ,
        "目标",
        data.pose_command?.online ? "#ff62cf" : "#7f8994",
        snapshotAnnotation(data.pose_command),
        true,
        true,
    );

    const actualZ = finiteNumber(data.tf?.data?.position_m?.z);
    drawDepthMarker(
        actualZ,
        "实际",
        data.tf?.online ? "#42e7a8" : "#8a97a6",
        snapshotAnnotation(data.tf),
    );

    ctx.fillStyle = "#8fb4ce";
    ctx.font = "bold 10px Microsoft YaHei, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Z / Down", width / 2, 4);
    ctx.fillText("正向 ↓", width / 2, height - 13);
}


function drawHeading(data) {
    const canvas = document.getElementById("heading-canvas");
    const {context: ctx, width, height} = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.max(25, Math.min(width, height) * 0.39);
    const tfOrientation = data.tf?.data?.orientation_deg || {};
    const heading = finiteNumber(tfOrientation.heading_deg);

    ctx.strokeStyle = "#42617b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();

    for (let angle = 0; angle < 360; angle += 15) {
        const radians = angle * Math.PI / 180;
        const major = angle % 45 === 0;
        const outerX = centerX + Math.sin(radians) * radius;
        const outerY = centerY - Math.cos(radians) * radius;
        const innerRadius = radius - (major ? 10 : 5);
        const innerX = centerX + Math.sin(radians) * innerRadius;
        const innerY = centerY - Math.cos(radians) * innerRadius;
        ctx.strokeStyle = major ? "#9cc8e4" : "#47657d";
        ctx.lineWidth = major ? 1.8 : 1;
        ctx.beginPath();
        ctx.moveTo(innerX, innerY);
        ctx.lineTo(outerX, outerY);
        ctx.stroke();
    }

    const cardinal = [
        ["N", 0, "#ff7c88"],
        ["E", 90, "#dceaf5"],
        ["S", 180, "#dceaf5"],
        ["W", 270, "#dceaf5"],
    ];
    ctx.font = "bold 12px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    cardinal.forEach(([label, angle, color]) => {
        const radians = angle * Math.PI / 180;
        ctx.fillStyle = color;
        ctx.fillText(
            label,
            centerX + Math.sin(radians) * (radius - 20),
            centerY - Math.cos(radians) * (radius - 20),
        );
    });

    if (heading !== null) {
        const radians = heading * Math.PI / 180;
        const tipX = centerX + Math.sin(radians) * (radius - 13);
        const tipY = centerY - Math.cos(radians) * (radius - 13);
        ctx.strokeStyle = data.tf?.online ? "#42e7a8" : "#8795a3";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(tipX, tipY);
        ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.fillStyle = "#eaf5fc";
    ctx.font = "bold 15px Consolas, monospace";
    ctx.fillText(numberText(heading, 1, "°"), centerX, centerY + radius * 0.48);

    if (!data.tf?.online) {
        ctx.fillStyle = "rgba(42, 48, 55, 0.68)";
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = "#d6dce2";
        ctx.font = "bold 12px Microsoft YaHei, sans-serif";
        ctx.fillText("姿态失效", centerX, centerY);
    }

    document.getElementById("heading-readout").textContent = [
        `TF Heading ${numberText(heading, 1, "°")}`,
        snapshotAnnotation(data.tf),
    ].filter(Boolean).join(" · ");
}


function drawHorizon(data) {
    const canvas = document.getElementById("horizon-canvas");
    const {context: ctx, width, height} = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.max(24, Math.min(width, height) * 0.41);
    const tfOrientation = data.tf?.data?.orientation_deg || {};
    const roll = finiteNumber(tfOrientation.roll_deg);
    const pitch = finiteNumber(tfOrientation.pitch_deg);
    const drawRoll = roll ?? 0;
    const drawPitch = Math.max(-45, Math.min(45, pitch ?? 0));
    const pixelsPerDegree = radius / 30;
    const span = radius * 3;

    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.clip();
    ctx.translate(centerX, centerY);
    ctx.rotate(-drawRoll * Math.PI / 180);
    ctx.translate(0, drawPitch * pixelsPerDegree);

    ctx.fillStyle = "#298ec4";
    ctx.fillRect(-span, -span, span * 2, span);
    ctx.fillStyle = "#8a5b32";
    ctx.fillRect(-span, 0, span * 2, span);

    ctx.strokeStyle = "#f8f0d7";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-span, 0);
    ctx.lineTo(span, 0);
    ctx.stroke();

    ctx.font = "9px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let mark = -30; mark <= 30; mark += 5) {
        if (mark === 0) continue;
        const y = -mark * pixelsPerDegree;
        const major = mark % 10 === 0;
        const lineHalf = major ? radius * 0.34 : radius * 0.20;
        ctx.strokeStyle = "rgba(255,255,255,0.86)";
        ctx.lineWidth = major ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(-lineHalf, y);
        ctx.lineTo(lineHalf, y);
        ctx.stroke();
        if (major) {
            ctx.fillStyle = "#fff";
            ctx.fillText(String(Math.abs(mark)), -lineHalf - 11, y);
            ctx.fillText(String(Math.abs(mark)), lineHalf + 11, y);
        }
    }
    ctx.restore();

    ctx.strokeStyle = "#55748d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = "#ffe07a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX - radius * 0.55, centerY);
    ctx.lineTo(centerX - radius * 0.16, centerY);
    ctx.lineTo(centerX, centerY + 6);
    ctx.lineTo(centerX + radius * 0.16, centerY);
    ctx.lineTo(centerX + radius * 0.55, centerY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#ffe07a";
    ctx.fill();

    if (!data.tf?.online) {
        ctx.fillStyle = "rgba(42, 48, 55, 0.70)";
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#d6dce2";
        ctx.font = "bold 12px Microsoft YaHei, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("姿态失效", centerX, centerY);
    }

    document.getElementById("horizon-readout").textContent = [
        `Roll ${numberText(roll, 1, "°")}`,
        `Pitch ${numberText(pitch, 1, "°")}`,
    ].join(" · ");
}


function drawNavigation(data) {
    drawXYMap(data);
    drawZAxis(data);
    drawHeading(data);
    drawHorizon(data);
}


function renderDashboard(data) {
    dashboardState.status = data;
    document.getElementById("server-time").textContent =
        `服务器时间 ${new Date(data.server_time * 1000).toLocaleString()}`;

    renderGlobalBadges(data);
    renderCamera("left", data.streams?.left);
    renderCamera("right", data.streams?.right);
    renderCamera("fisheye", data.streams?.fisheye);

    const ready = document.getElementById("status-ready");
    ready.textContent = data.ready ? "坐标系已就绪" : "坐标系未就绪";
    ready.className = `ready-label ${data.ready ? "online" : "offline"}`;

    renderCoreStatus(data);
    renderMotionState(data);
    renderMotionDiagnostics(data);
    renderActuatorStatus(data);
    renderPowerStatus(data);
    renderSystemStatus(data);

    drawNavigation(data);
}


async function refreshStatus() {
    try {
        const response = await fetch("/api/status", {cache: "no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        dashboardState.connected = true;
        renderDashboard(data);
    } catch (error) {
        dashboardState.connected = false;
        const timeElement = document.getElementById("server-time");
        timeElement.textContent = `Web 状态连接失败：${error.message}`;
        const container = document.getElementById("global-badges");
        container.replaceChildren(badge("Web 连接失败", false));
    }
}


function configureMapInteraction() {
    const canvas = document.getElementById("xy-canvas");
    const zCanvas = document.getElementById("z-canvas");

    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0012);
        dashboardState.mapScale = Math.max(
            4,
            Math.min(420, dashboardState.mapScale * factor),
        );
        if (dashboardState.status) drawNavigation(dashboardState.status);
    }, {passive: false});

    canvas.addEventListener("pointerdown", (event) => {
        dashboardState.dragging = true;
        dashboardState.dragStartX = event.clientX;
        dashboardState.dragStartY = event.clientY;
        dashboardState.dragPanX = dashboardState.mapPanX;
        dashboardState.dragPanY = dashboardState.mapPanY;
        canvas.classList.add("is-dragging");
        canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
        if (!dashboardState.dragging) return;
        dashboardState.mapPanX = (
            dashboardState.dragPanX
            + event.clientX
            - dashboardState.dragStartX
        );
        dashboardState.mapPanY = (
            dashboardState.dragPanY
            + event.clientY
            - dashboardState.dragStartY
        );
        if (dashboardState.status) drawNavigation(dashboardState.status);
    });

    const stopDragging = (event) => {
        dashboardState.dragging = false;
        canvas.classList.remove("is-dragging");
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
    };
    canvas.addEventListener("pointerup", stopDragging);
    canvas.addEventListener("pointercancel", stopDragging);

    zCanvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0012);
        const previousScale = dashboardState.zScale;
        const nextScale = Math.max(
            4,
            Math.min(420, previousScale * factor),
        );
        const rect = zCanvas.getBoundingClientRect();
        const pointerY = event.clientY - rect.top;
        const previousCenterY = rect.height / 2 + dashboardState.zPanY;
        const pointerDepth = (
            (pointerY - previousCenterY)
            / previousScale
        );

        dashboardState.zScale = nextScale;
        dashboardState.zPanY = (
            pointerY
            - pointerDepth * nextScale
            - rect.height / 2
        );
        if (dashboardState.status) drawZAxis(dashboardState.status);
    }, {passive: false});

    zCanvas.addEventListener("pointerdown", (event) => {
        dashboardState.zDragging = true;
        dashboardState.zDragStartY = event.clientY;
        dashboardState.zDragPanY = dashboardState.zPanY;
        zCanvas.classList.add("is-dragging");
        zCanvas.setPointerCapture(event.pointerId);
    });

    zCanvas.addEventListener("pointermove", (event) => {
        if (!dashboardState.zDragging) return;
        dashboardState.zPanY = (
            dashboardState.zDragPanY
            + event.clientY
            - dashboardState.zDragStartY
        );
        if (dashboardState.status) drawZAxis(dashboardState.status);
    });

    const stopZDragging = (event) => {
        dashboardState.zDragging = false;
        zCanvas.classList.remove("is-dragging");
        if (zCanvas.hasPointerCapture(event.pointerId)) {
            zCanvas.releasePointerCapture(event.pointerId);
        }
    };
    zCanvas.addEventListener("pointerup", stopZDragging);
    zCanvas.addEventListener("pointercancel", stopZDragging);

    document.getElementById("reset-map").addEventListener("click", () => {
        dashboardState.mapScale = 20;
        dashboardState.mapPanX = 0;
        dashboardState.mapPanY = 0;
        dashboardState.zScale = 20;
        dashboardState.zPanY = 0;
        if (dashboardState.status) drawNavigation(dashboardState.status);
    });
}


function initialize() {
    configureMapInteraction();
    window.addEventListener("resize", () => {
        if (dashboardState.status) drawNavigation(dashboardState.status);
    });
    refreshStatus();
    window.setInterval(refreshStatus, 250);
}


initialize();
