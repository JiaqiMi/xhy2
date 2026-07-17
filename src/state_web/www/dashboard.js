"use strict";

const dashboardState = {
    status: null,
    connected: false,
    trajectory: [],
    originRevision: null,
    lastTrackStamp: null,
    mapScale: 20,
    mapPanX: 0,
    mapPanY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragPanX: 0,
    dragPanY: 0,
};

const MAX_TRAJECTORY_POINTS = 12000;


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
    if (!snapshot || !snapshot.data || !snapshot.online) return "bad";
    return "good";
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


function renderPoseStatus(data) {
    const tfPose = data.tf?.data || {};
    const tfPosition = tfPose.position_m || {};
    const tfOrientation = tfPose.orientation_deg || {};
    const feedback = data.feedback?.data || {};
    const feedbackPose = feedback.pose || {};
    const attitude = data.attitude?.actual || {};

    setRows("pose-status", [
        {
            label: "TF 状态",
            value: snapshotText(data.tf),
            className: snapshotClass(data.tf),
        },
        {label: "X / North", value: numberText(tfPosition.x, 3, " m")},
        {label: "Y / East", value: numberText(tfPosition.y, 3, " m")},
        {label: "Z / Down", value: numberText(tfPosition.z, 3, " m")},
        {label: "TF Roll", value: numberText(tfOrientation.roll_deg, 2, "°")},
        {label: "TF Pitch", value: numberText(tfOrientation.pitch_deg, 2, "°")},
        {label: "TF Heading", value: numberText(tfOrientation.heading_deg, 2, "°")},
        {
            label: "姿态来源",
            value: attitude.source || "--",
            className: data.attitude?.valid ? "good" : "bad",
        },
        {
            label: "AUV 反馈",
            value: snapshotText(data.feedback),
            className: snapshotClass(data.feedback),
        },
        {label: "纬度", value: numberText(feedbackPose.latitude_deg, 7, "°")},
        {label: "经度", value: numberText(feedbackPose.longitude_deg, 7, "°")},
        {label: "反馈深度", value: numberText(feedbackPose.depth_m, 3, " m")},
        {label: "反馈 Roll", value: numberText(feedbackPose.roll_deg, 2, "°")},
        {label: "反馈 Pitch", value: numberText(feedbackPose.pitch_deg, 2, "°")},
        {label: "反馈 Heading", value: numberText(feedbackPose.heading_deg, 2, "°")},
        {
            label: "控制模式",
            value: feedback.control_mode_name
                ? `${feedback.control_mode_name} (${feedback.control_mode})`
                : "--",
        },
    ]);
}


function renderVelocityStatus(data) {
    const velocity = data.velocity?.data || {};
    const linear = velocity.linear_mps || {};
    const angular = velocity.angular_radps || {};

    setRows("velocity-status", [
        {
            label: "速度话题",
            value: snapshotText(data.velocity),
            className: snapshotClass(data.velocity),
        },
        {label: "坐标系", value: velocity.frame_id || "--"},
        {label: "线速度 X", value: numberText(linear.x, 3, " m/s")},
        {label: "线速度 Y", value: numberText(linear.y, 3, " m/s")},
        {label: "线速度 Z", value: numberText(linear.z, 3, " m/s")},
        {label: "角速度 X", value: numberText(angular.x, 3, " rad/s")},
        {label: "角速度 Y", value: numberText(angular.y, 3, " rad/s")},
        {label: "角速度 Z", value: numberText(angular.z, 3, " rad/s")},
        {label: "角速度 X", value: numberText(radToDeg(angular.x), 2, "°/s")},
        {label: "角速度 Y", value: numberText(radToDeg(angular.y), 2, "°/s")},
        {label: "角速度 Z", value: numberText(radToDeg(angular.z), 2, "°/s")},
    ]);
}


function renderPoseCommand(data) {
    const command = data.pose_command?.data || {};
    const target = command.target || {};
    const position = target.position_m || {};
    const orientation = target.orientation_deg || {};
    const force = command.force || {};

    setRows("pose-command-status", [
        {
            label: "指令状态",
            value: snapshotText(data.pose_command),
            className: snapshotClass(data.pose_command),
        },
        {
            label: "模式",
            value: command.mode_name
                ? `${command.mode_name} (${command.mode})`
                : "--",
        },
        {label: "目标 X", value: numberText(position.x, 3, " m")},
        {label: "目标 Y", value: numberText(position.y, 3, " m")},
        {label: "目标 Z", value: numberText(position.z, 3, " m")},
        {label: "目标 Roll", value: numberText(orientation.roll_deg, 2, "°")},
        {label: "目标 Pitch", value: numberText(orientation.pitch_deg, 2, "°")},
        {label: "目标 Heading", value: numberText(orientation.heading_deg, 2, "°")},
        {label: "TX", value: integerText(force.tx)},
        {label: "TY", value: integerText(force.ty)},
        {label: "TZ", value: integerText(force.tz)},
        {label: "MX", value: integerText(force.mx)},
        {label: "MY", value: integerText(force.my)},
        {label: "MZ", value: integerText(force.mz)},
    ]);
}


function renderMotionState(data) {
    const motion = data.motion_state?.data || {};
    const goal = motion.goal || {};
    const goalPosition = goal.position_m || {};
    const goalOrientation = goal.orientation_deg || {};
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
        {label: "目标 X", value: numberText(goalPosition.x, 3, " m")},
        {label: "目标 Y", value: numberText(goalPosition.y, 3, " m")},
        {label: "目标 Z", value: numberText(goalPosition.z, 3, " m")},
        {label: "目标航向", value: numberText(goalOrientation.heading_deg, 2, "°")},
        {label: "输出 TX", value: integerText(force.tx)},
        {label: "输出 TY", value: integerText(force.ty)},
        {label: "输出 MZ", value: integerText(force.mz)},
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


function updateTrajectory(data) {
    const revision = finiteNumber(data.origin?.data?.revision);
    if (
        revision !== null
        && dashboardState.originRevision !== null
        && revision !== dashboardState.originRevision
    ) {
        dashboardState.trajectory = [];
        dashboardState.lastTrackStamp = null;
    }
    if (revision !== null) dashboardState.originRevision = revision;

    const enabled = document.getElementById("trajectory-enabled").checked;
    const tfPose = data.tf?.data;
    const position = tfPose?.position_m;
    if (!enabled || !data.tf?.online || !position) return;

    const stamp = data.tf.ros_stamp ?? data.tf.received_at;
    if (stamp === null || stamp === undefined || stamp === dashboardState.lastTrackStamp) {
        return;
    }

    const point = {
        x: finiteNumber(position.x),
        y: finiteNumber(position.y),
        z: finiteNumber(position.z),
        stamp,
    };
    if ([point.x, point.y, point.z].some((value) => value === null)) return;

    dashboardState.trajectory.push(point);
    dashboardState.lastTrackStamp = stamp;
    if (dashboardState.trajectory.length > MAX_TRAJECTORY_POINTS) {
        dashboardState.trajectory.splice(
            0,
            dashboardState.trajectory.length - MAX_TRAJECTORY_POINTS,
        );
    }
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

    if (dashboardState.trajectory.length > 1) {
        ctx.strokeStyle = "#23c6f4";
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.82;
        ctx.beginPath();
        dashboardState.trajectory.forEach((point, index) => {
            const screen = worldToScreen(point.x, point.y);
            if (index === 0) ctx.moveTo(screen.x, screen.y);
            else ctx.lineTo(screen.x, screen.y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    const position = data.tf?.data?.position_m;
    if (position) {
        const north = finiteNumber(position.x);
        const east = finiteNumber(position.y);
        if (north !== null && east !== null) {
            const screen = worldToScreen(north, east);
            const actual = data.attitude?.actual;
            const heading = finiteNumber(actual?.heading_deg);
            const online = Boolean(data.tf?.online);

            ctx.fillStyle = online ? "#42e7a8" : "#8a97a6";
            ctx.strokeStyle = "#031018";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(screen.x, screen.y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            if (heading !== null) {
                const radians = heading * Math.PI / 180;
                const arrowLength = 28;
                const tipX = screen.x + Math.sin(radians) * arrowLength;
                const tipY = screen.y - Math.cos(radians) * arrowLength;
                ctx.strokeStyle = online ? "#42e7a8" : "#8a97a6";
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(screen.x, screen.y);
                ctx.lineTo(tipX, tipY);
                ctx.stroke();
            }

            ctx.fillStyle = "#d9ecfa";
            ctx.font = "11px Consolas, monospace";
            ctx.fillText(
                `N ${numberText(north, 2)}  E ${numberText(east, 2)}`,
                screen.x + 10,
                screen.y + 8,
            );
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

    if (!data.tf?.online) {
        ctx.fillStyle = "rgba(7, 12, 19, 0.45)";
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = "#ff9aa6";
        ctx.font = "bold 13px Microsoft YaHei, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("TF 位姿无效或已超时", width / 2, 18);
        ctx.textAlign = "left";
    }
}


function drawZAxis(data) {
    const canvas = document.getElementById("z-canvas");
    const {context: ctx, width, height} = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    const scale = dashboardState.mapScale;
    const centerY = height / 2;
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

    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = "#23c6f4";
    ctx.lineWidth = 1;
    dashboardState.trajectory.forEach((point) => {
        const screenY = centerY + point.z * scale;
        if (screenY >= 0 && screenY <= height) {
            ctx.beginPath();
            ctx.moveTo(axisX - 15, screenY);
            ctx.lineTo(axisX + 15, screenY);
            ctx.stroke();
        }
    });
    ctx.globalAlpha = 1;

    const z = finiteNumber(data.tf?.data?.position_m?.z);
    if (z !== null) {
        const screenY = centerY + z * scale;
        const clampedY = Math.max(12, Math.min(height - 12, screenY));
        ctx.strokeStyle = data.tf?.online ? "#42e7a8" : "#8a97a6";
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(8, clampedY);
        ctx.lineTo(width - 6, clampedY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(8, clampedY);
        ctx.lineTo(17, clampedY - 6);
        ctx.lineTo(17, clampedY + 6);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = "#e4f1fb";
        ctx.font = "bold 10px Consolas, monospace";
        ctx.fillText(`Z ${numberText(z, 2)}`, 4, Math.max(2, clampedY - 19));
    }

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
    const actual = data.attitude?.actual;
    const target = data.attitude?.target;
    const heading = finiteNumber(actual?.heading_deg);

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
        ctx.strokeStyle = data.attitude?.valid ? "#42e7a8" : "#8795a3";
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

    if (target?.valid && finiteNumber(target.heading_deg) !== null) {
        const radians = Number(target.heading_deg) * Math.PI / 180;
        const markerRadius = radius + 1;
        const markerX = centerX + Math.sin(radians) * markerRadius;
        const markerY = centerY - Math.cos(radians) * markerRadius;
        ctx.save();
        ctx.translate(markerX, markerY);
        ctx.rotate(-radians);
        ctx.fillStyle = "#ff62cf";
        ctx.beginPath();
        ctx.moveTo(0, 8);
        ctx.lineTo(-6, -5);
        ctx.lineTo(6, -5);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    ctx.fillStyle = "#eaf5fc";
    ctx.font = "bold 15px Consolas, monospace";
    ctx.fillText(numberText(heading, 1, "°"), centerX, centerY + radius * 0.48);

    if (!data.attitude?.valid) {
        ctx.fillStyle = "rgba(42, 48, 55, 0.68)";
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = "#d6dce2";
        ctx.font = "bold 12px Microsoft YaHei, sans-serif";
        ctx.fillText("姿态失效", centerX, centerY);
    }

    document.getElementById("heading-readout").textContent = [
        `实际 ${numberText(heading, 1, "°")}`,
        `目标 ${target?.valid ? numberText(target.heading_deg, 1, "°") : "--"}`,
        `误差 ${numberText(data.attitude?.heading_error_deg, 1, "°")}`,
    ].join(" · ");
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
    const actual = data.attitude?.actual || {};
    const roll = finiteNumber(actual.roll_deg);
    const pitch = finiteNumber(actual.pitch_deg);
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

    if (!data.attitude?.valid) {
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

    renderPoseStatus(data);
    renderVelocityStatus(data);
    renderPoseCommand(data);
    renderMotionState(data);
    renderActuatorStatus(data);
    renderPowerStatus(data);
    renderSystemStatus(data);

    updateTrajectory(data);
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
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
    };
    canvas.addEventListener("pointerup", stopDragging);
    canvas.addEventListener("pointercancel", stopDragging);

    document.getElementById("clear-trajectory").addEventListener("click", () => {
        dashboardState.trajectory = [];
        dashboardState.lastTrackStamp = null;
        if (dashboardState.status) drawNavigation(dashboardState.status);
    });

    document.getElementById("reset-map").addEventListener("click", () => {
        dashboardState.mapScale = 20;
        dashboardState.mapPanX = 0;
        dashboardState.mapPanY = 0;
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
