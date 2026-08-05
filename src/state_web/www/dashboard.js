"use strict";

/*
名称：dashboard.js
功能：state_web 状态渲染、相机状态与导航地图交互
作者：xhy
监听：Web 状态接口与用户交互
发布：仪表盘绘制及浏览器地图朝向、水池范围设置
记录：
2026.7.30
    增加可持久化的手动地图旋转，支持指定航向为上。
    增加拖拽对角点绘制并持久化水池 N/E 边界。
    水池矩形改为按绘制时地图航向确定方向，并保存世界坐标角点。
2026.8.2
    在左目和鱼眼标题旁显示各在线视觉任务的检测帧率。
2026.8.3
    新增视觉地图绘制、持久化显示开关、历史清除和鱼眼识别历史。
    新增 base_link 一分钟轨迹、独立清除按钮和最近两帧目标绘制。
    新增视觉位置与置信度文字的独立持久化显示开关。
    增加 XY 位置图双击全页放大、再次双击原位恢复功能。
    在实际 base_link 到 camera 箭头上叠加无标注的深绿色 hand 点。
    将最新和上一目标及其连接线改为高对比粉色系。
    清除操作完成后恢复单行工具栏使用的短按钮名称。
    将视觉识别 arrow 改为青蓝色，与粉色 target 明确区分。
    增加只锁定 base_link 居中的实时跟踪，不自动旋转地图航向。
2026.8.5
    增加执行器实际反馈图形、6S4P 电池摘要和鱼眼裁切/全图放大交互。
    航向仪表增加粉色目标指针，详细状态只保留执行器反馈。
    执行器改为浅色指令与深色反馈叠图，并显示接收年龄、接收差和同步状态。
    执行器改为浅色指令与深色反馈叠图，并显示接收年龄、接收差和同步状态。
*/

const MAP_UP_HEADING_KEY = "state_web.map_up_heading_deg";
const POOL_BOUNDARY_KEY = "state_web.pool_boundary_ned";
const VISUAL_HISTORY_VISIBLE_KEY = "state_web.visual_history_visible";
const VISUAL_LABELS_VISIBLE_KEY = "state_web.visual_labels_visible";


function loadVisualHistoryVisible() {
    try {
        const saved = window.localStorage.getItem(
            VISUAL_HISTORY_VISIBLE_KEY,
        );
        return saved === null ? true : saved !== "false";
    } catch (error) {
        return true;
    }
}


function saveVisualHistoryVisible(visible) {
    try {
        window.localStorage.setItem(
            VISUAL_HISTORY_VISIBLE_KEY,
            String(Boolean(visible)),
        );
    } catch (error) {
        // 浏览器禁用本地存储时，本次页面内开关仍然有效。
    }
}


function loadVisualLabelsVisible() {
    try {
        const saved = window.localStorage.getItem(
            VISUAL_LABELS_VISIBLE_KEY,
        );
        return saved === null ? true : saved !== "false";
    } catch (error) {
        return true;
    }
}


function saveVisualLabelsVisible(visible) {
    try {
        window.localStorage.setItem(
            VISUAL_LABELS_VISIBLE_KEY,
            String(Boolean(visible)),
        );
    } catch (error) {
        // 浏览器禁用本地存储时，本次页面内开关仍然有效。
    }
}


function normalizeMapHeading(value) {
    const heading = Number(value);
    if (!Number.isFinite(heading)) return null;
    return ((heading % 360) + 360) % 360;
}


function loadMapUpHeading() {
    try {
        return normalizeMapHeading(
            window.localStorage.getItem(MAP_UP_HEADING_KEY),
        ) ?? 0;
    } catch (error) {
        return 0;
    }
}


function saveMapUpHeading(heading) {
    try {
        window.localStorage.setItem(
            MAP_UP_HEADING_KEY,
            String(heading),
        );
    } catch (error) {
        // 浏览器禁用本地存储时，本次页面内设置仍然有效。
    }
}


function mapPointToWorld(mapNorth, mapEast, heading) {
    const rotation = heading * Math.PI / 180;
    const rotationCos = Math.cos(rotation);
    const rotationSin = Math.sin(rotation);
    return {
        east: mapEast * rotationCos + mapNorth * rotationSin,
        north: -mapEast * rotationSin + mapNorth * rotationCos,
    };
}


function normalizePoolBoundary(value) {
    if (!value || typeof value !== "object") return null;

    if (Array.isArray(value.corners) && value.corners.length === 4) {
        const corners = value.corners.map((point) => ({
            north: finiteNumber(point?.north),
            east: finiteNumber(point?.east),
        }));
        if (corners.some((point) => (
            point.north === null || point.east === null
        ))) {
            return null;
        }
        const heading = normalizeMapHeading(value.headingDeg);
        if (heading === null) return null;
        const widthM = Math.hypot(
            corners[1].north - corners[0].north,
            corners[1].east - corners[0].east,
        );
        const lengthM = Math.hypot(
            corners[2].north - corners[1].north,
            corners[2].east - corners[1].east,
        );
        return {
            headingDeg: heading,
            corners,
            lengthM,
            widthM,
        };
    }

    // 兼容旧版固定 N/E 边界，迁移后按 0° 北向上处理。
    const northMin = finiteNumber(value.northMin);
    const northMax = finiteNumber(value.northMax);
    const eastMin = finiteNumber(value.eastMin);
    const eastMax = finiteNumber(value.eastMax);
    if ([northMin, northMax, eastMin, eastMax].includes(null)) return null;
    const normalizedNorthMin = Math.min(northMin, northMax);
    const normalizedNorthMax = Math.max(northMin, northMax);
    const normalizedEastMin = Math.min(eastMin, eastMax);
    const normalizedEastMax = Math.max(eastMin, eastMax);
    return {
        headingDeg: 0,
        corners: [
            { north: normalizedNorthMin, east: normalizedEastMin },
            { north: normalizedNorthMin, east: normalizedEastMax },
            { north: normalizedNorthMax, east: normalizedEastMax },
            { north: normalizedNorthMax, east: normalizedEastMin },
        ],
        lengthM: normalizedNorthMax - normalizedNorthMin,
        widthM: normalizedEastMax - normalizedEastMin,
    };
}


function poolBoundaryFromMapPoints(first, second, heading) {
    if (!first || !second) return null;
    const mapNorthMin = Math.min(first.north, second.north);
    const mapNorthMax = Math.max(first.north, second.north);
    const mapEastMin = Math.min(first.east, second.east);
    const mapEastMax = Math.max(first.east, second.east);
    return normalizePoolBoundary({
        headingDeg: heading,
        corners: [
            mapPointToWorld(mapNorthMin, mapEastMin, heading),
            mapPointToWorld(mapNorthMin, mapEastMax, heading),
            mapPointToWorld(mapNorthMax, mapEastMax, heading),
            mapPointToWorld(mapNorthMax, mapEastMin, heading),
        ],
    });
}


function loadPoolBounds() {
    try {
        const saved = window.localStorage.getItem(POOL_BOUNDARY_KEY);
        return saved ? normalizePoolBoundary(JSON.parse(saved)) : null;
    } catch (error) {
        return null;
    }
}


function savePoolBounds(bounds) {
    try {
        if (bounds) {
            window.localStorage.setItem(
                POOL_BOUNDARY_KEY,
                JSON.stringify(bounds),
            );
        } else {
            window.localStorage.removeItem(POOL_BOUNDARY_KEY);
        }
    } catch (error) {
        // 浏览器禁用本地存储时，本次页面内设置仍然有效。
    }
}


const dashboardState = {
    status: null,
    connected: false,
    mapScale: 20,
    mapPanX: 0,
    mapPanY: 0,
    mapTracking: false,
    mapUpHeading: loadMapUpHeading(),
    expandedCamera: null,
    visualHistoryVisible: loadVisualHistoryVisible(),
    visualLabelsVisible: loadVisualLabelsVisible(),
    poolBounds: loadPoolBounds(),
    poolDraftBounds: null,
    poolDrawing: false,
    poolDrawStartMap: null,
    poolDrawHeading: 0,
    poolDrawStartClientX: 0,
    poolDrawStartClientY: 0,
    zScale: 20,
    zPanY: 0,
    xyMapExpanded: false,
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
    element.className = `badge ${alarm ? "alarm" : (warning ? "warning" : (online ? "online" : "offline"))
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
        const labels = { left: "左目", right: "右目", fisheye: "鱼眼" };
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


function renderVisionFps(vision) {
    const labels = {
        line: "线",
        red_circle: "红圆",
        shapes: "形状",
        rectangle: "方框",
        arrow: "箭头",
        aruco: "ArUco",
    };
    for (const camera of ["left", "fisheye"]) {
        const element = document.getElementById(
            `camera-${camera}-vision-fps`,
        );
        const items = Object.entries(vision || {})
            .filter(([, source]) => source?.camera === camera)
            .map(([name, source]) => {
                const channel = source?.channels?.detection;
                if (!channel?.online) return null;
                const fps = finiteNumber(source.fps ?? channel.fps);
                const fpsText = fps !== null && fps > 0
                    ? fps.toFixed(1)
                    : "--";
                return `${labels[name] || source.label || name} ${fpsText} FPS`;
            })
            .filter(Boolean);
        element.textContent = items.length
            ? items.join(" · ")
            : "视觉 FPS --";
        element.classList.toggle("is-online", items.length > 0);
    }
}


function arucoAgeText(value) {
    const age = finiteNumber(value);
    if (age === null) return "--前";
    if (age < 1) return `${Math.max(0, age).toFixed(1)}秒前`;
    if (age < 10) return `${age.toFixed(1)}秒前`;
    return `${Math.round(age)}秒前`;
}


function renderArucoHistory(history) {
    const container = document.getElementById("aruco-history");
    const expected = document.getElementById("aruco-expected-color");
    const items = Array.isArray(history?.items) ? history.items : [];
    const fragment = document.createDocumentFragment();
    if (!items.length) {
        const empty = document.createElement("span");
        empty.className = "aruco-history-empty";
        empty.textContent = "暂无有效识别";
        fragment.appendChild(empty);
    } else {
        items.slice(0, 10).forEach((item) => {
            const marker = document.createElement("span");
            marker.className = "aruco-history-item";
            marker.textContent = `ID ${integerText(item.marker_id)} · ${arucoAgeText(item.age_sec)}`;
            marker.title = `置信度 ${numberText(item.confidence, 2)}`;
            fragment.appendChild(marker);
        });
    }
    container.replaceChildren(fragment);

    const colors = {
        yellow: "黄色",
        green: "绿色",
        red: "红色",
    };
    const color = colors[history?.expected_color]
        ? history.expected_color
        : null;
    expected.className = `aruco-expected-color ${color || "pending"}`;
    expected.querySelector(".aruco-color-text").textContent = color
        ? `期望${colors[color]}`
        : "待确认";
    expected.title = color
        ? `锁存 ID ${integerText(history.confirmed_marker_id)}，当前窗口命中 ${integerText(history.confirmed_count)}/${integerText(history.required_count)}`
        : `最近 ${integerText(history?.window_size)} 次中尚无 ID 达到 ${integerText(history?.required_count)} 次`;
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
                { axis: "X / North", value: numberText(tfPosition.x, 3, " m"), className: tfClass },
                { axis: "Y / East", value: numberText(tfPosition.y, 3, " m"), className: tfClass },
                { axis: "Z / Down", value: numberText(tfPosition.z, 3, " m"), className: tfClass },
            ],
        },
        {
            label: "目标位置",
            cells: [
                { axis: "X / North", value: numberText(targetPosition.x, 3, " m"), className: commandClass },
                { axis: "Y / East", value: numberText(targetPosition.y, 3, " m"), className: commandClass },
                { axis: "Z / Down", value: numberText(targetPosition.z, 3, " m"), className: commandClass },
            ],
        },
        {
            label: "位置误差",
            title: "目标位置减实际 TF 位置",
            cells: [
                { axis: "ΔX", value: numberText(numericDifference(targetPosition.x, tfPosition.x), 3, " m"), className: poseErrorClass },
                { axis: "ΔY", value: numberText(numericDifference(targetPosition.y, tfPosition.y), 3, " m"), className: poseErrorClass },
                { axis: "ΔZ", value: numberText(numericDifference(targetPosition.z, tfPosition.z), 3, " m"), className: poseErrorClass },
            ],
        },
        {
            label: "实际姿态",
            cells: [
                { axis: "Roll", value: numberText(tfOrientation.roll_deg, 2, "°"), className: tfClass },
                { axis: "Pitch", value: numberText(tfOrientation.pitch_deg, 2, "°"), className: tfClass },
                { axis: "Heading", value: numberText(tfOrientation.heading_deg, 2, "°"), className: tfClass },
            ],
        },
        {
            label: "目标姿态",
            cells: [
                { axis: "Roll", value: numberText(targetOrientation.roll_deg, 2, "°"), className: commandClass },
                { axis: "Pitch", value: numberText(targetOrientation.pitch_deg, 2, "°"), className: commandClass },
                { axis: "Heading", value: numberText(targetOrientation.heading_deg, 2, "°"), className: commandClass },
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
                { axis: "TX / MX", value: `${integerText(actualForce.tx)} / ${integerText(actualForce.mx)}`, className: feedbackClass },
                { axis: "TY / MY", value: `${integerText(actualForce.ty)} / ${integerText(actualForce.my)}`, className: feedbackClass },
                { axis: "TZ / MZ", value: `${integerText(actualForce.tz)} / ${integerText(actualForce.mz)}`, className: feedbackClass },
            ],
        },
        {
            label: "目标力 / 力矩",
            title: "cmdned 指令；每列依次显示平移力 T 与旋转力矩 M",
            cells: [
                { axis: "TX / MX", value: `${integerText(targetForce.tx)} / ${integerText(targetForce.mx)}`, className: commandClass },
                { axis: "TY / MY", value: `${integerText(targetForce.ty)} / ${integerText(targetForce.my)}`, className: commandClass },
                { axis: "TZ / MZ", value: `${integerText(targetForce.tz)} / ${integerText(targetForce.mz)}`, className: commandClass },
            ],
        },
        {
            label: "线速度",
            cells: [
                { axis: "X", value: numberText(linear.x, 3, " m/s"), className: velocityClass },
                { axis: "Y", value: numberText(linear.y, 3, " m/s"), className: velocityClass },
                { axis: "Z", value: numberText(linear.z, 3, " m/s"), className: velocityClass },
            ],
        },
        {
            label: "角速度",
            cells: [
                { axis: "X", value: numberText(radToDeg(angular.x), 2, "°/s"), className: velocityClass },
                { axis: "Y", value: numberText(radToDeg(angular.y), 2, "°/s"), className: velocityClass },
                { axis: "Z", value: numberText(radToDeg(angular.z), 2, "°/s"), className: velocityClass },
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
        { label: "位置误差", value: numberText(motion.position_error_m, 3, " m") },
        { label: "航向误差", value: numberText(radToDeg(motion.yaw_error_rad), 2, "°") },
        { label: "水平速度", value: numberText(motion.horizontal_speed_mps, 3, " m/s") },
        { label: "航向角速度", value: numberText(radToDeg(motion.yaw_rate_radps), 2, "°/s") },
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


function gripperDescription(value) {
    const number = finiteNumber(value);
    if (number === null) return "未知";
    const clamped = Math.max(0, Math.min(255, number));
    if (clamped <= 0) return "全开";
    if (clamped >= 255) return "全闭";
    return `开度 ${Math.round((255 - clamped) * 100 / 255)}%`;
}


function setGripperGeometry(prefix, value, online) {
    const number = online ? finiteNumber(value) : null;
    const clamped = number === null ? 127.5 : Math.max(0, Math.min(255, number));
    const jawGap = 8 + (255 - clamped) * 52 / 255;
    const leftJawX = 80 - jawGap / 2;
    const rightJawX = 80 + jawGap / 2;
    ["left-arm", "left-finger"].forEach((suffix) => {
        document.getElementById(`gripper-${prefix}-${suffix}`).setAttribute(
            "x2", leftJawX.toFixed(1),
        );
    });
    ["right-arm", "right-finger"].forEach((suffix) => {
        document.getElementById(`gripper-${prefix}-${suffix}`).setAttribute(
            "x2", rightJawX.toFixed(1),
        );
    });
    document.getElementById(`gripper-${prefix}-left-finger`).setAttribute(
        "x1", leftJawX.toFixed(1),
    );
    document.getElementById(`gripper-${prefix}-right-finger`).setAttribute(
        "x1", rightJawX.toFixed(1),
    );
    document.querySelector(`.actuator-${prefix}-shape`).classList.toggle(
        "is-unknown", !online,
    );
    return number === null ? null : clamped;
}


function pushrodDescription(value) {
    const command = finiteNumber(value);
    if (command === 0) return "停止";
    if (command === 1) return "前进";
    if (command === 2) return "反转";
    return "未知";
}


function setPushrodLane(prefix, commandValue, speedValue, online) {
    const command = online ? finiteNumber(commandValue) : null;
    const speed = online ? finiteNumber(speedValue) : null;
    const clampedSpeed = speed === null ? null : Math.max(0, Math.min(255, speed));
    const motionLine = document.getElementById(`pushrod-${prefix}-motion-line`);
    const stopMarker = document.getElementById(`pushrod-${prefix}-stop-marker`);
    if (command === 0) {
        motionLine.style.display = "none";
        stopMarker.style.display = "block";
    } else if (command === 1 || command === 2) {
        const forward = command === 1;
        motionLine.setAttribute("x1", forward ? "51" : "125");
        motionLine.setAttribute("x2", forward ? "125" : "51");
        motionLine.style.display = "block";
        stopMarker.style.display = "none";
    } else {
        motionLine.style.display = "none";
        stopMarker.style.display = "none";
    }
    document.getElementById(`pushrod-${prefix}-speed-bar`).setAttribute(
        "width",
        clampedSpeed === null ? "0" : (124 * clampedSpeed / 255).toFixed(1),
    );
    document.getElementById("pushrod-state-text").textContent = driveText;
    document.getElementById("pushrod-value").textContent = clampedSpeed === null
        ? "速度 --"
        : `速度 ${Math.round(clampedSpeed)} / 255`;

    setRows("actuator-status", [
        {
            label: "执行同步",
            value: stateText,
            className: synchronized ? "good" : (commandOnline || feedbackOnline ? "warning" : "bad"),
            title: "接收差为 Web 收到两类消息的时间差，不等同于硬件执行延迟",
        },
        { label: "指令年龄", value: ageText(command.actuator_age_sec) },
        { label: "反馈年龄", value: ageText(data.actuator_feedback?.age_sec) },
        { label: "当前接收差", value: ageText(receiveGap) },
        { label: "首次匹配接收差", value: ageText(acknowledgedGap) },
        { label: "最后指令模式", value: command.last_mode_name || "--" },
        { label: "反馈模式", value: feedback.mode_name || "--" },
        { label: "补光灯1 指/馈", value: `${integerText(command.light1)} / ${integerText(feedback.light1)}` },
        { label: "补光灯2 指/馈", value: `${integerText(command.light2)} / ${integerText(feedback.light2)}` },
        { label: "航向舵机 指/馈", value: `${integerText(command.heading_servo)} / ${integerText(feedback.heading_servo)}` },
        { label: "夹爪 指/馈", value: `${integerText(commandClamp)} / ${integerText(feedbackClamp)}` },
        { label: "推杆动作 指/馈", value: `${commandDrive.text} / ${feedbackDrive.text}` },
        { label: "推杆速度 指/馈", value: `${integerText(commandDrive.speed)} / ${integerText(feedbackDrive.speed)}` },
        { label: "红黄绿 指/馈", value: `${integerText(command.red_light)}${integerText(command.yellow_light)}${integerText(command.green_light)} / ${integerText(feedback.red_light)}${integerText(feedback.yellow_light)}${integerText(feedback.green_light)}` },
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
            label: "XY 锁存 / 进出次数",
            value: diagnostics.xy_brake_latched === undefined ? "--" : (
                `${diagnostics.xy_brake_latched ? "锁存中" : "未锁存"} / `
                + `${integerText(diagnostics.xy_brake_entry_count)} / ${integerText(diagnostics.xy_brake_exit_count)}`
            ),
            className: diagnostics.xy_brake_latched ? "warning" : "good",
        },
        {
            label: "Yaw 速度参考 / 停止角",
            value: `${numberText(radToDeg(diagnostics.yaw_rate_reference), 2)} °/s / ${numberText(radToDeg(diagnostics.yaw_stop_angle), 2)} °`,
        },
        {
            label: "地图航向角速度",
            value: `${numberText(radToDeg(diagnostics.map_yaw_rate), 2)} °/s`,
        },
        {
            label: "Yaw 主动制动",
            value: diagnostics.yaw_braking === undefined ? "--" : (diagnostics.yaw_braking ? "进行中" : "跟踪中"),
            className: diagnostics.yaw_braking ? "warning" : "good",
        },
        {
            label: "Yaw 锁存 / 进出次数",
            value: diagnostics.yaw_brake_latched === undefined ? "--" : (
                `${diagnostics.yaw_brake_latched ? "锁存中" : "未锁存"} / `
                + `${integerText(diagnostics.yaw_brake_entry_count)} / ${integerText(diagnostics.yaw_brake_exit_count)}`
            ),
            className: diagnostics.yaw_brake_latched ? "warning" : "good",
        },
        {
            label: "当前制动轴",
            value: diagnostics.brake_axes || "无",
            className: diagnostics.brake_axes ? "warning" : "good",
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
    const summary = power.summary || {};
    const battery = summary.battery || {};
    const sensor = data.feedback?.data?.sensor || {};
    const leak = sensor.leak_alarm;
    const fault = finiteNumber(sensor.fault_status);
    const online = Boolean(data.power?.online && summary.valid);
    const powerState = document.getElementById("power-summary-state");
    powerState.textContent = online
        ? `在线 · ${ageText(data.power?.age_sec)}`
        : "电源离线";
    powerState.className = `compact-state ${online ? "online" : "offline"}`;

    const setMetric = (id, value, digits, suffix) => {
        document.getElementById(id).textContent = online
            ? numberText(value, digits, suffix)
            : "--";
    };
    setMetric("power-battery-voltage", summary.battery_voltage_v, 2, " V");
    setMetric("power-control-current", summary.control_current_a, 2, " A");
    setMetric("power-control-power", summary.control_power_w, 1, " W");
    setMetric("power-motive-current", summary.motive_current_a, 2, " A");
    setMetric("power-motive-power", summary.motive_power_w, 1, " W");
    setMetric("power-total-power", summary.total_power_w, 1, " W");

    const soc = online && battery.valid
        ? finiteNumber(battery.soc_percent)
        : null;
    const remaining = online && battery.valid
        ? finiteNumber(battery.remaining_ah)
        : null;
    const capacity = finiteNumber(battery.pack_capacity_ah) ?? 16.0;
    const level = document.getElementById("battery-level");
    const clampedSoc = soc === null ? 0 : Math.max(0, Math.min(100, soc));
    level.style.width = `${clampedSoc.toFixed(1)}%`;
    level.className = `battery-level ${soc === null
        ? "unknown"
        : (soc < 20 ? "low" : (soc < 40 ? "medium" : "good"))}`;
    document.getElementById("battery-percent").textContent = soc === null
        ? "估算电量 --"
        : `估算电量 ${soc.toFixed(1)}%`;
    document.getElementById("battery-capacity").textContent = remaining === null
        ? `-- / ${capacity.toFixed(2)} Ah`
        : `${remaining.toFixed(2)} / ${capacity.toFixed(2)} Ah`;

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
        { label: "动力电压", value: numberText(summary.battery_voltage_v, 2, " V") },
        { label: "控制电流", value: numberText(summary.control_current_a, 2, " A") },
        { label: "控制功率", value: numberText(summary.control_power_w, 2, " W") },
        { label: "动力电流", value: numberText(summary.motive_current_a, 2, " A") },
        { label: "动力功率", value: numberText(summary.motive_power_w, 2, " W") },
        { label: "总功率", value: numberText(summary.total_power_w, 2, " W") },
        { label: "5秒平滑电压", value: numberText(battery.smoothed_voltage_v, 2, " V") },
        { label: "单节估算电压", value: numberText(battery.cell_voltage_v, 3, " V") },
        { label: "估算电量", value: numberText(battery.soc_percent, 1, "%") },
        { label: "剩余 / 总容量", value: `${numberText(battery.remaining_ah, 2)} / ${numberText(battery.pack_capacity_ah, 2, " Ah")}` },
        { label: "舱内温度", value: numberText(sensor.temperature_c, 1, " ℃") },
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
        { label: "传感器有效位", value: hexadecimal(sensor.sensor_valid, 2) },
        { label: "传感器更新位", value: hexadecimal(sensor.sensor_updated, 2) },
        { label: "设备电源位", value: hexadecimal(sensor.power_status, 4) },
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
        { label: "世界坐标系", value: data.frames?.world || "--" },
        { label: "机器人坐标系", value: data.frames?.base || "--" },
        { label: "原点版本", value: integerText(origin.revision) },
        { label: "原点纬度", value: numberText(origin.latitude_deg, 7, "°") },
        { label: "原点经度", value: numberText(origin.longitude_deg, 7, "°") },
        { label: "原点深度", value: numberText(origin.depth_m, 3, " m") },
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
    return { context, width, height };
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
    const { base, camera } = points;

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


function drawHandPoint(ctx, screen) {
    if (!screen) return;
    ctx.save();
    ctx.fillStyle = "#087844";
    ctx.strokeStyle = "#031018";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}


function clipLineToCanvas(start, end, width, height) {
    const deltaX = end.x - start.x;
    const deltaY = end.y - start.y;
    let minimum = 0;
    let maximum = 1;
    const boundaries = [
        [-deltaX, start.x],
        [deltaX, width - start.x],
        [-deltaY, start.y],
        [deltaY, height - start.y],
    ];

    for (const [direction, distance] of boundaries) {
        if (Math.abs(direction) < 1e-9) {
            if (distance < 0) return null;
            continue;
        }
        const ratio = distance / direction;
        if (direction < 0) {
            minimum = Math.max(minimum, ratio);
        } else {
            maximum = Math.min(maximum, ratio);
        }
        if (minimum > maximum) return null;
    }

    return [
        {
            x: start.x + minimum * deltaX,
            y: start.y + minimum * deltaY,
        },
        {
            x: start.x + maximum * deltaX,
            y: start.y + maximum * deltaY,
        },
    ];
}


function drawMapCompass(ctx, width, upHeading) {
    const center = { x: Math.max(42, width - 48), y: 50 };
    const arrowLength = 24;
    const axes = [
        { label: "N", heading: -upHeading, color: "#43c7ff" },
        { label: "E", heading: 90 - upHeading, color: "#ffbe45" },
    ];

    ctx.save();
    ctx.font = "bold 11px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const axis of axes) {
        const radians = axis.heading * Math.PI / 180;
        const tip = {
            x: center.x + Math.sin(radians) * arrowLength,
            y: center.y - Math.cos(radians) * arrowLength,
        };
        ctx.strokeStyle = axis.color;
        ctx.fillStyle = axis.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(center.x, center.y);
        ctx.lineTo(tip.x, tip.y);
        ctx.stroke();
        ctx.fillText(
            axis.label,
            tip.x + Math.sin(radians) * 9,
            tip.y - Math.cos(radians) * 9,
        );
    }
    ctx.fillStyle = "#d9e8f5";
    ctx.beginPath();
    ctx.arc(center.x, center.y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "10px Microsoft YaHei, Consolas, monospace";
    ctx.fillText(
        `上 ${numberText(upHeading, upHeading % 1 ? 1 : 0, "°")}`,
        center.x,
        center.y + 39,
    );
    ctx.restore();
}


function createMapTransform(width, height) {
    const scale = dashboardState.mapScale;
    const originX = width / 2 + dashboardState.mapPanX;
    const originY = height / 2 + dashboardState.mapPanY;
    const rotation = dashboardState.mapUpHeading * Math.PI / 180;
    const rotationCos = Math.cos(rotation);
    const rotationSin = Math.sin(rotation);
    const mapToWorld = (mapNorth, mapEast) => ({
        east: mapEast * rotationCos + mapNorth * rotationSin,
        north: -mapEast * rotationSin + mapNorth * rotationCos,
    });
    const screenToMap = (screenX, screenY) => ({
        east: (screenX - originX) / scale,
        north: (originY - screenY) / scale,
    });

    return {
        scale,
        worldToScreen(north, east) {
            const mapEast = east * rotationCos - north * rotationSin;
            const mapNorth = east * rotationSin + north * rotationCos;
            return {
                x: originX + mapEast * scale,
                y: originY - mapNorth * scale,
            };
        },
        mapToWorld,
        screenToMap,
        screenToWorld(screenX, screenY) {
            const point = screenToMap(screenX, screenY);
            return mapToWorld(point.north, point.east);
        },
    };
}


function updateBaseTrackingPan(data) {
    if (!dashboardState.mapTracking) return false;
    const tfData = data?.tf?.data || {};
    const basePose = tfData.frame_poses?.base || tfData;
    const north = finiteNumber(basePose.position_m?.x);
    const east = finiteNumber(basePose.position_m?.y);
    if (north === null || east === null) return false;

    const rotation = dashboardState.mapUpHeading * Math.PI / 180;
    const rotationCos = Math.cos(rotation);
    const rotationSin = Math.sin(rotation);
    const mapEast = east * rotationCos - north * rotationSin;
    const mapNorth = east * rotationSin + north * rotationCos;
    dashboardState.mapPanX = -mapEast * dashboardState.mapScale;
    dashboardState.mapPanY = mapNorth * dashboardState.mapScale;
    return true;
}


function drawPoolBoundary(ctx, worldToScreen, boundary, draft = false) {
    if (!boundary) return;
    const corners = boundary.corners.map((point) => (
        worldToScreen(point.north, point.east)
    ));
    const centerWorld = boundary.corners.reduce(
        (center, point) => ({
            north: center.north + point.north / 4,
            east: center.east + point.east / 4,
        }),
        { north: 0, east: 0 },
    );
    const positiveWorld = {
        north: (
            boundary.corners[2].north + boundary.corners[3].north
        ) / 2,
        east: (
            boundary.corners[2].east + boundary.corners[3].east
        ) / 2,
    };
    const center = worldToScreen(centerWorld.north, centerWorld.east);
    const positive = worldToScreen(
        positiveWorld.north,
        positiveWorld.east,
    );

    ctx.save();
    ctx.fillStyle = draft
        ? "rgba(255, 190, 69, 0.12)"
        : "rgba(67, 199, 255, 0.10)";
    ctx.strokeStyle = draft ? "#ffbe45" : "#43c7ff";
    ctx.lineWidth = draft ? 2.5 : 2;
    ctx.setLineDash(draft ? [7, 5] : []);
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let index = 1; index < corners.length; index += 1) {
        ctx.lineTo(corners[index].x, corners[index].y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = draft ? "#ffbe45" : "#43c7ff";
    for (const corner of corners) {
        ctx.beginPath();
        ctx.arc(corner.x, corner.y, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    // 从矩形中心指向绘制时的“地图上方”，明确水池正方向。
    const directionX = positive.x - center.x;
    const directionY = positive.y - center.y;
    const directionLength = Math.hypot(directionX, directionY);
    if (directionLength >= 6) {
        const unitX = directionX / directionLength;
        const unitY = directionY / directionLength;
        const normalX = -unitY;
        const normalY = unitX;
        const headLength = 9;
        const headWidth = 5;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(center.x, center.y);
        ctx.lineTo(positive.x, positive.y);
        ctx.lineTo(
            positive.x - unitX * headLength + normalX * headWidth,
            positive.y - unitY * headLength + normalY * headWidth,
        );
        ctx.moveTo(positive.x, positive.y);
        ctx.lineTo(
            positive.x - unitX * headLength - normalX * headWidth,
            positive.y - unitY * headLength - normalY * headWidth,
        );
        ctx.stroke();
    }

    const label = [
        draft ? "水池范围（绘制中）" : "水池范围",
        `正向 ${numberText(boundary.headingDeg, 0, "°")}`,
        `中心 N ${numberText(centerWorld.north, 2)}`,
        `E ${numberText(centerWorld.east, 2)}`,
        `${numberText(boundary.lengthM, 2)} × ${numberText(boundary.widthM, 2)} m`,
    ].join(" · ");
    ctx.font = "bold 11px Microsoft YaHei, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(3, 16, 24, 0.95)";
    ctx.strokeText(label, center.x, center.y - 8);
    ctx.fillText(label, center.x, center.y - 8);
    ctx.restore();
}


function drawBaseTrajectory(ctx, worldToScreen, trajectory) {
    const points = Array.isArray(trajectory?.points)
        ? trajectory.points.map((point) => ({
            north: finiteNumber(point?.north_m),
            east: finiteNumber(point?.east_m),
        })).filter((point) => (
            point.north !== null && point.east !== null
        ))
        : [];
    if (!points.length) return;

    const screens = points.map((point) => (
        worldToScreen(point.north, point.east)
    ));
    ctx.save();
    ctx.strokeStyle = "#48a9ff";
    ctx.fillStyle = "#48a9ff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (screens.length > 1) {
        ctx.beginPath();
        ctx.moveTo(screens[0].x, screens[0].y);
        screens.slice(1).forEach((point) => {
            ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
    }
    screens.forEach((point, index) => {
        ctx.globalAlpha = 0.32 + 0.68 * (index + 1) / screens.length;
        ctx.beginPath();
        ctx.arc(point.x, point.y, index === screens.length - 1 ? 3.5 : 2.3,
            0, Math.PI * 2);
        ctx.fill();
    });
    ctx.restore();
}


function targetHistoryItems(data) {
    const history = Array.isArray(data.target_history?.items)
        ? data.target_history.items.slice(0, 2)
        : [];
    if (history.length) return history;
    const current = data.pose_command?.data?.target;
    return current ? [current] : [];
}


const VISUAL_MAP_STYLES = {
    red_circle: { color: "#ff3f50", marker: "circle" },
    black_square: { color: "#080b10", outline: "#e8f0fa", marker: "square" },
    yellow_circle: { color: "#ffd642", marker: "circle" },
    red_line: { color: "#ff5364", marker: "line" },
    arrow: { color: "#22d3ee", marker: "arrow" },
    rectangle_red: { color: "#ff5364", marker: "circle" },
    rectangle_yellow: { color: "#ffd642", marker: "circle" },
    rectangle_green: { color: "#38d996", marker: "circle" },
};


function rectanglesOverlap(first, second) {
    return !(
        first.x + first.width < second.x
        || second.x + second.width < first.x
        || first.y + first.height < second.y
        || second.y + second.height < first.y
    );
}


function drawVisualMapLabel(
    ctx,
    anchor,
    text,
    color,
    occupied,
    width,
    height,
) {
    ctx.save();
    ctx.font = "bold 12px Microsoft YaHei, Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const textWidth = Math.ceil(ctx.measureText(text).width);
    const labelWidth = textWidth + 10;
    const labelHeight = 20;
    const offsets = [
        [10, -25], [10, 8], [-labelWidth - 10, -25],
        [-labelWidth - 10, 8], [10, -47], [-labelWidth - 10, -47],
    ];
    let box = null;
    for (const [offsetX, offsetY] of offsets) {
        const candidate = {
            x: Math.max(2, Math.min(width - labelWidth - 2, anchor.x + offsetX)),
            y: Math.max(2, Math.min(height - labelHeight - 2, anchor.y + offsetY)),
            width: labelWidth,
            height: labelHeight,
        };
        if (!occupied.some((item) => rectanglesOverlap(candidate, item))) {
            box = candidate;
            break;
        }
        if (box === null) box = candidate;
    }
    occupied.push(box);
    ctx.fillStyle = "rgba(3, 13, 22, 0.82)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = color;
    ctx.fillText(text, box.x + 5, box.y + 3);
    ctx.restore();
}


function drawVisualPointMarker(ctx, screen, style) {
    ctx.save();
    ctx.fillStyle = style.color;
    ctx.strokeStyle = style.outline || "rgba(3, 12, 19, 0.95)";
    ctx.lineWidth = style.outline ? 2 : 1.5;
    if (style.marker === "square") {
        ctx.fillRect(screen.x - 6, screen.y - 6, 12, 12);
        ctx.strokeRect(screen.x - 6, screen.y - 6, 12, 12);
    } else {
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    ctx.restore();
}


function drawVisualArrow(ctx, origin, directionTip, color) {
    let directionX = directionTip.x - origin.x;
    let directionY = directionTip.y - origin.y;
    const length = Math.hypot(directionX, directionY);
    if (length <= 1e-6) return;
    directionX /= length;
    directionY /= length;
    const arrowLength = 30;
    const tip = {
        x: origin.x + directionX * arrowLength,
        y: origin.y + directionY * arrowLength,
    };
    const normalX = -directionY;
    const normalY = directionX;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(tip.x, tip.y);
    ctx.lineTo(
        tip.x - directionX * 9 + normalX * 5,
        tip.y - directionY * 9 + normalY * 5,
    );
    ctx.moveTo(tip.x, tip.y);
    ctx.lineTo(
        tip.x - directionX * 9 - normalX * 5,
        tip.y - directionY * 9 - normalY * 5,
    );
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
}


function drawVisualMapHistory(ctx, worldToScreen, visionMap, width, height) {
    if (!dashboardState.visualHistoryVisible) return;
    const categories = visionMap?.categories;
    if (!categories || typeof categories !== "object") return;
    const occupied = [];
    for (const [category, style] of Object.entries(VISUAL_MAP_STYLES)) {
        const records = Array.isArray(categories[category])
            ? categories[category]
            : [];
        records.forEach((record, index) => {
            const points = Array.isArray(record?.points)
                ? record.points.map((point) => ({
                    north: finiteNumber(point?.north_m),
                    east: finiteNumber(point?.east_m),
                })).filter((point) => (
                    point.north !== null && point.east !== null
                ))
                : [];
            if (!points.length) return;
            const screens = points.map((point) => (
                worldToScreen(point.north, point.east)
            ));
            const confidence = finiteNumber(record?.confidence);
            const alpha = 0.48 + 0.52 * (index + 1) / Math.max(1, records.length);
            ctx.save();
            ctx.globalAlpha = alpha;
            if (style.marker === "line") {
                if (screens.length < 2) {
                    ctx.restore();
                    return;
                }
                ctx.strokeStyle = style.color;
                ctx.fillStyle = style.color;
                ctx.lineWidth = 2.5;
                ctx.lineJoin = "round";
                ctx.beginPath();
                ctx.moveTo(screens[0].x, screens[0].y);
                screens.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
                ctx.stroke();
                screens.forEach((point) => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
                    ctx.fill();
                });
            } else if (style.marker === "arrow") {
                const direction = record?.direction_ne;
                const north = finiteNumber(direction?.north);
                const east = finiteNumber(direction?.east);
                if (north === null || east === null) {
                    ctx.restore();
                    return;
                }
                drawVisualArrow(
                    ctx,
                    screens[0],
                    worldToScreen(
                        points[0].north + north,
                        points[0].east + east,
                    ),
                    style.color,
                );
            } else {
                drawVisualPointMarker(ctx, screens[0], style);
            }
            ctx.restore();

            if (!dashboardState.visualLabelsVisible) return;
            const anchor = screens[Math.floor((screens.length - 1) / 2)];
            const confidenceText = confidence === null
                ? "C --"
                : `C ${confidence.toFixed(2)}`;
            const label = style.marker === "line"
                ? confidenceText
                : `${confidenceText} · N ${points[0].north.toFixed(2)} E ${points[0].east.toFixed(2)}`;
            drawVisualMapLabel(
                ctx,
                anchor,
                label,
                style.outline || style.color,
                occupied,
                width,
                height,
            );
        });
    }
}


function drawXYMap(data) {
    const canvas = document.getElementById("xy-canvas");
    const { context: ctx, width, height } = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    updateBaseTrackingPan(data);
    const mapTransform = createMapTransform(width, height);
    const { scale, worldToScreen, screenToWorld } = mapTransform;
    const upHeading = dashboardState.mapUpHeading;
    const gridStep = niceDistance(70 / scale);
    const visibleCorners = [
        screenToWorld(0, 0),
        screenToWorld(width, 0),
        screenToWorld(0, height),
        screenToWorld(width, height),
    ];
    const eastValues = visibleCorners.map((point) => point.east);
    const northValues = visibleCorners.map((point) => point.north);
    const eastMin = Math.min(...eastValues);
    const eastMax = Math.max(...eastValues);
    const northMin = Math.min(...northValues);
    const northMax = Math.max(...northValues);

    ctx.lineWidth = 1;
    ctx.font = "10px Consolas, monospace";
    ctx.textBaseline = "top";

    for (
        let east = Math.ceil(eastMin / gridStep) * gridStep;
        east <= eastMax + gridStep * 0.5;
        east += gridStep
    ) {
        const line = clipLineToCanvas(
            worldToScreen(northMin - gridStep, east),
            worldToScreen(northMax + gridStep, east),
            width,
            height,
        );
        if (!line) continue;
        const isAxis = Math.abs(east) < gridStep * 0.01;
        ctx.strokeStyle = isAxis ? "#3c799c" : "#19354a";
        ctx.beginPath();
        ctx.moveTo(line[0].x, line[0].y);
        ctx.lineTo(line[1].x, line[1].y);
        ctx.stroke();
        const labelPoint = line[0].y <= line[1].y ? line[0] : line[1];
        ctx.fillStyle = "#66849d";
        ctx.fillText(
            numberText(east, gridStep < 1 ? 1 : 0),
            Math.max(3, Math.min(width - 36, labelPoint.x + 3)),
            Math.max(3, Math.min(height - 14, labelPoint.y + 3)),
        );
    }

    for (
        let north = Math.ceil(northMin / gridStep) * gridStep;
        north <= northMax + gridStep * 0.5;
        north += gridStep
    ) {
        const line = clipLineToCanvas(
            worldToScreen(north, eastMin - gridStep),
            worldToScreen(north, eastMax + gridStep),
            width,
            height,
        );
        if (!line) continue;
        const isAxis = Math.abs(north) < gridStep * 0.01;
        ctx.strokeStyle = isAxis ? "#3c799c" : "#19354a";
        ctx.beginPath();
        ctx.moveTo(line[0].x, line[0].y);
        ctx.lineTo(line[1].x, line[1].y);
        ctx.stroke();
        const labelPoint = line[0].x <= line[1].x ? line[0] : line[1];
        ctx.fillStyle = "#66849d";
        ctx.fillText(
            numberText(north, gridStep < 1 ? 1 : 0),
            Math.max(3, Math.min(width - 36, labelPoint.x + 3)),
            Math.max(3, Math.min(height - 14, labelPoint.y + 3)),
        );
    }

    const poolBounds = dashboardState.poolDraftBounds
        || dashboardState.poolBounds;
    drawPoolBoundary(
        ctx,
        worldToScreen,
        poolBounds,
        Boolean(dashboardState.poolDraftBounds),
    );
    drawBaseTrajectory(ctx, worldToScreen, data.base_trajectory);
    drawVisualMapHistory(
        ctx,
        worldToScreen,
        data.vision_map,
        width,
        height,
    );

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
    const mapActualHeading = actualHeading === null
        ? null
        : normalizeMapHeading(actualHeading - upHeading);
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
        hand: frameScreen(framePoses.hand),
    };
    const hasActualFrameArrow = Boolean(
        actualFramePoints.base
        && actualFramePoints.camera
    );

    const targetFrames = targetHistoryItems(data).map((target, index) => {
        const targetNorth = finiteNumber(target?.position_m?.x);
        const targetEast = finiteNumber(target?.position_m?.y);
        const targetHeading = finiteNumber(
            target?.orientation_deg?.heading_deg,
        );
        return {
            target,
            index,
            north: targetNorth,
            east: targetEast,
            screen: targetNorth !== null && targetEast !== null
                ? worldToScreen(targetNorth, targetEast)
                : null,
            heading: targetHeading === null
                ? null
                : normalizeMapHeading(targetHeading - upHeading),
        };
    }).filter((frame) => frame.screen);
    const latestTarget = targetFrames[0] || null;

    if (actualScreen && latestTarget) {
        ctx.save();
        ctx.strokeStyle = "#ff78cf";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([5, 5]);
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.moveTo(actualScreen.x, actualScreen.y);
        ctx.lineTo(latestTarget.screen.x, latestTarget.screen.y);
        ctx.stroke();
        ctx.restore();
    }

    [...targetFrames].reverse().forEach((frame) => {
        const latest = frame.index === 0;
        const annotation = latest
            ? snapshotAnnotation(data.pose_command)
            : `收到于 ${ageText(frame.target?.age_sec)}`;
        drawDirectionalPose(ctx, frame.screen, frame.heading, {
            color: latest ? "#ff4fbd" : "#ffb0df",
            label: [
                `${latest ? "最新目标" : "上一目标"} base_link N ${numberText(frame.north, 2)}  E ${numberText(frame.east, 2)}`,
                annotation,
            ].filter(Boolean).join(" · "),
            marker: "diamond",
            labelOffsetY: latest ? -25 : 11,
        });
    });

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
                mapActualHeading,
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
            drawDirectionalPose(ctx, actualScreen, mapActualHeading, {
                color: actualColor,
                label: actualLabel,
            });
        }
        drawHandPoint(ctx, actualFramePoints.hand);
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

    drawMapCompass(ctx, width, upHeading);

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
    const { context: ctx, width, height } = resizeCanvas(canvas);
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

    [...targetHistoryItems(data)].reverse().forEach((target, reverseIndex, all) => {
        const latest = reverseIndex === all.length - 1;
        drawDepthMarker(
            finiteNumber(target?.position_m?.z),
            latest ? "最新目标" : "上一目标",
            latest ? "#ff4fbd" : "#ffb0df",
            latest
                ? snapshotAnnotation(data.pose_command)
                : `收到于 ${ageText(target?.age_sec)}`,
            true,
            latest,
        );
    });

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
    const { context: ctx, width, height } = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07111d";
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.max(25, Math.min(width, height) * 0.39);
    const tfOrientation = data.tf?.data?.orientation_deg || {};
    const heading = finiteNumber(tfOrientation.heading_deg);
    const targetHeading = data.attitude?.target?.valid
        ? finiteNumber(data.attitude.target.heading_deg)
        : null;

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

    if (targetHeading !== null) {
        const radians = targetHeading * Math.PI / 180;
        const tipX = centerX + Math.sin(radians) * (radius - 17);
        const tipY = centerY - Math.cos(radians) * (radius - 17);
        ctx.save();
        ctx.strokeStyle = "#ff62cf";
        ctx.fillStyle = "#ff62cf";
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(tipX, tipY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.translate(tipX, tipY);
        ctx.rotate(targetHeading * Math.PI / 180);
        ctx.beginPath();
        ctx.moveTo(0, -7);
        ctx.lineTo(6, 4);
        ctx.lineTo(-6, 4);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

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

    ctx.save();
    ctx.font = "bold 9px Microsoft YaHei, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#42e7a8";
    ctx.fillText("实", 6, 5);
    ctx.fillStyle = "#ff62cf";
    ctx.fillText("目", 24, 5);
    ctx.restore();

    if (!data.tf?.online) {
        ctx.fillStyle = "rgba(42, 48, 55, 0.68)";
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = "#d6dce2";
        ctx.font = "bold 12px Microsoft YaHei, sans-serif";
        ctx.fillText("姿态失效", centerX, centerY);
    }

    document.getElementById("heading-readout").textContent = [
        `实际 ${numberText(heading, 1, "°")}`,
        `目标 ${numberText(targetHeading, 1, "°")}`,
    ].join(" · ");
}


function drawHorizon(data) {
    const canvas = document.getElementById("horizon-canvas");
    const { context: ctx, width, height } = resizeCanvas(canvas);
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
    renderVisionFps(data.vision);
    renderArucoHistory(data.aruco_history);

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
        const response = await fetch("/api/status", { cache: "no-store" });
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


function mapHeadingText() {
    const heading = dashboardState.mapUpHeading;
    return heading % 1
        ? heading.toFixed(1)
        : heading.toFixed(0);
}


function updateMapHint() {
    const hint = document.getElementById("map-hint");
    if (dashboardState.poolDrawing) {
        hint.textContent = dashboardState.poolDrawStartMap
            ? `松开完成水池矩形 · 上方 ${mapHeadingText()}°`
            : `按住并拖拽水池两个对角点 · 上方 ${mapHeadingText()}°`;
    } else {
        const expansionText = dashboardState.xyMapExpanded
            ? "双击恢复"
            : "双击放大";
        hint.textContent =
            `滚轮缩放 · 拖拽平移 · ${expansionText} · 上方 ${mapHeadingText()}°`;
    }
}


function updateMapHeadingControls() {
    const headingText = mapHeadingText();
    document.getElementById("map-up-heading").value = headingText;
    updateMapHint();
}


function configureMapHeading() {
    const input = document.getElementById("map-up-heading");
    const applyButton = document.getElementById("apply-map-heading");
    const northUpButton = document.getElementById("north-up-map");

    const redraw = () => {
        updateMapHeadingControls();
        if (dashboardState.status) drawXYMap(dashboardState.status);
    };
    const applyInput = () => {
        const heading = finiteNumber(input.value);
        if (
            heading === null
            || !Number.isInteger(heading)
            || heading < 0
            || heading > 359
        ) {
            input.setCustomValidity("请输入 0 到 359 之间的整数航向");
            input.reportValidity();
            return;
        }
        input.setCustomValidity("");
        dashboardState.mapUpHeading = normalizeMapHeading(heading);
        saveMapUpHeading(dashboardState.mapUpHeading);
        redraw();
    };

    input.addEventListener("input", () => input.setCustomValidity(""));
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") applyInput();
    });
    applyButton.addEventListener("click", applyInput);
    northUpButton.addEventListener("click", () => {
        dashboardState.mapUpHeading = 0;
        saveMapUpHeading(0);
        redraw();
    });
    updateMapHeadingControls();
}


function updatePoolBoundaryControls() {
    const canvas = document.getElementById("xy-canvas");
    const drawButton = document.getElementById("draw-pool-boundary");
    const clearButton = document.getElementById("clear-pool-boundary");
    drawButton.textContent = dashboardState.poolDrawing
        ? "取消绘制"
        : (dashboardState.poolBounds ? "重画水池" : "绘制水池");
    drawButton.classList.toggle(
        "is-active",
        dashboardState.poolDrawing,
    );
    clearButton.disabled = !(
        dashboardState.poolBounds
        || dashboardState.poolDraftBounds
    );
    canvas.classList.toggle("is-drawing", dashboardState.poolDrawing);
    updateMapHint();
}


function configurePoolBoundary() {
    const drawButton = document.getElementById("draw-pool-boundary");
    const clearButton = document.getElementById("clear-pool-boundary");

    drawButton.addEventListener("click", () => {
        dashboardState.poolDrawing = !dashboardState.poolDrawing;
        dashboardState.poolDraftBounds = null;
        dashboardState.poolDrawStartMap = null;
        updatePoolBoundaryControls();
        if (dashboardState.status) drawXYMap(dashboardState.status);
    });
    clearButton.addEventListener("click", () => {
        dashboardState.poolBounds = null;
        dashboardState.poolDraftBounds = null;
        dashboardState.poolDrawing = false;
        dashboardState.poolDrawStartMap = null;
        savePoolBounds(null);
        updatePoolBoundaryControls();
        if (dashboardState.status) drawXYMap(dashboardState.status);
    });
    updatePoolBoundaryControls();
}


function configureVisualHistoryControls() {
    const toggle = document.getElementById("show-visual-history");
    const labelToggle = document.getElementById("show-visual-labels");
    const clearButton = document.getElementById("clear-visual-history");
    toggle.checked = dashboardState.visualHistoryVisible;
    labelToggle.checked = dashboardState.visualLabelsVisible;
    toggle.addEventListener("change", () => {
        dashboardState.visualHistoryVisible = toggle.checked;
        saveVisualHistoryVisible(toggle.checked);
        if (dashboardState.status) drawXYMap(dashboardState.status);
    });
    labelToggle.addEventListener("change", () => {
        dashboardState.visualLabelsVisible = labelToggle.checked;
        saveVisualLabelsVisible(labelToggle.checked);
        if (dashboardState.status) drawXYMap(dashboardState.status);
    });

    clearButton.addEventListener("click", async () => {
        if (clearButton.disabled) return;
        clearButton.disabled = true;
        clearButton.textContent = "清除中…";
        try {
            const response = await fetch("/api/vision-history/clear", {
                method: "POST",
                cache: "no-store",
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            await response.json();
            clearButton.textContent = "已清除";
            await refreshStatus();
        } catch (error) {
            clearButton.textContent = "清除失败";
            clearButton.title = `清除视觉历史失败：${error.message}`;
        } finally {
            window.setTimeout(() => {
                clearButton.disabled = false;
                clearButton.textContent = "清视觉";
                clearButton.title = "清除地图视觉绘图、鱼眼历史和期望颜色";
            }, 1200);
        }
    });
}


function configureBaseTrajectoryControl() {
    const clearButton = document.getElementById("clear-base-trajectory");
    clearButton.addEventListener("click", async () => {
        if (clearButton.disabled) return;
        clearButton.disabled = true;
        clearButton.textContent = "清除中…";
        try {
            const response = await fetch("/api/base-trajectory/clear", {
                method: "POST",
                cache: "no-store",
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            await response.json();
            clearButton.textContent = "已清除";
            await refreshStatus();
        } catch (error) {
            clearButton.textContent = "清除失败";
            clearButton.title = `清除 base_link 轨迹失败：${error.message}`;
        } finally {
            window.setTimeout(() => {
                clearButton.disabled = false;
                clearButton.textContent = "清轨迹";
                clearButton.title = "只清除 base_link 轨迹";
            }, 1200);
        }
    });
}


function configureMapInteraction() {
    const canvas = document.getElementById("xy-canvas");
    const card = canvas.closest(".xy-card");
    const zCanvas = document.getElementById("z-canvas");
    const trackingButton = document.getElementById("track-base-link");
    const pointerMap = (event) => {
        const rect = canvas.getBoundingClientRect();
        const transform = createMapTransform(rect.width, rect.height);
        return transform.screenToMap(
            event.clientX - rect.left,
            event.clientY - rect.top,
        );
    };
    const redrawXYMap = () => {
        drawXYMap(dashboardState.status || {});
    };
    const updateTrackingControl = () => {
        trackingButton.classList.toggle(
            "is-active",
            dashboardState.mapTracking,
        );
        trackingButton.textContent = dashboardState.mapTracking
            ? "跟踪中"
            : "跟踪";
        trackingButton.setAttribute(
            "aria-pressed",
            String(dashboardState.mapTracking),
        );
        canvas.classList.toggle(
            "is-tracking",
            dashboardState.mapTracking,
        );
    };

    trackingButton.addEventListener("click", () => {
        dashboardState.mapTracking = !dashboardState.mapTracking;
        dashboardState.dragging = false;
        canvas.classList.remove("is-dragging");
        updateTrackingControl();
        redrawXYMap();
    });
    updateTrackingControl();
    const setMapExpanded = (expanded) => {
        dashboardState.xyMapExpanded = Boolean(expanded);
        card.classList.toggle(
            "is-expanded",
            dashboardState.xyMapExpanded,
        );
        document.body.classList.toggle(
            "xy-map-expanded",
            dashboardState.xyMapExpanded,
        );
        canvas.setAttribute(
            "aria-label",
            dashboardState.xyMapExpanded
                ? "NED XY 位置图，双击恢复原布局"
                : "NED XY 位置图，双击放大查看",
        );
        updateMapHint();
        window.requestAnimationFrame(() => {
            window.requestAnimationFrame(redrawXYMap);
        });
    };

    canvas.addEventListener("dblclick", (event) => {
        event.preventDefault();
        if (dashboardState.poolDrawing) return;
        setMapExpanded(!dashboardState.xyMapExpanded);
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && dashboardState.xyMapExpanded) {
            setMapExpanded(false);
        }
    });

    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        if (dashboardState.poolDrawing) return;
        const factor = Math.exp(-event.deltaY * 0.0012);
        dashboardState.mapScale = Math.max(
            4,
            Math.min(420, dashboardState.mapScale * factor),
        );
        if (dashboardState.status) drawNavigation(dashboardState.status);
    }, { passive: false });

    canvas.addEventListener("pointerdown", (event) => {
        if (dashboardState.poolDrawing) {
            dashboardState.poolDrawStartMap = pointerMap(event);
            dashboardState.poolDrawHeading = dashboardState.mapUpHeading;
            dashboardState.poolDrawStartClientX = event.clientX;
            dashboardState.poolDrawStartClientY = event.clientY;
            dashboardState.poolDraftBounds = poolBoundaryFromMapPoints(
                dashboardState.poolDrawStartMap,
                dashboardState.poolDrawStartMap,
                dashboardState.poolDrawHeading,
            );
            canvas.setPointerCapture(event.pointerId);
            updatePoolBoundaryControls();
            redrawXYMap();
            return;
        }
        if (dashboardState.mapTracking) return;
        dashboardState.dragging = true;
        dashboardState.dragStartX = event.clientX;
        dashboardState.dragStartY = event.clientY;
        dashboardState.dragPanX = dashboardState.mapPanX;
        dashboardState.dragPanY = dashboardState.mapPanY;
        canvas.classList.add("is-dragging");
        canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
        if (
            dashboardState.poolDrawing
            && dashboardState.poolDrawStartMap
        ) {
            dashboardState.poolDraftBounds = poolBoundaryFromMapPoints(
                dashboardState.poolDrawStartMap,
                pointerMap(event),
                dashboardState.poolDrawHeading,
            );
            redrawXYMap();
            return;
        }
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

    const stopDragging = (event, cancelled = false) => {
        if (
            dashboardState.poolDrawing
            && dashboardState.poolDrawStartMap
        ) {
            const dragDistance = Math.hypot(
                event.clientX - dashboardState.poolDrawStartClientX,
                event.clientY - dashboardState.poolDrawStartClientY,
            );
            const bounds = cancelled
                ? null
                : poolBoundaryFromMapPoints(
                    dashboardState.poolDrawStartMap,
                    pointerMap(event),
                    dashboardState.poolDrawHeading,
                );
            const hasArea = bounds
                && bounds.lengthM > 1e-6
                && bounds.widthM > 1e-6;
            if (dragDistance >= 4 && hasArea) {
                dashboardState.poolBounds = bounds;
                dashboardState.poolDrawing = false;
                savePoolBounds(bounds);
            }
            dashboardState.poolDraftBounds = null;
            dashboardState.poolDrawStartMap = null;
            if (canvas.hasPointerCapture(event.pointerId)) {
                canvas.releasePointerCapture(event.pointerId);
            }
            updatePoolBoundaryControls();
            redrawXYMap();
            return;
        }
        dashboardState.dragging = false;
        canvas.classList.remove("is-dragging");
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
    };
    canvas.addEventListener("pointerup", stopDragging);
    canvas.addEventListener(
        "pointercancel",
        (event) => stopDragging(event, true),
    );

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
    }, { passive: false });

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


function configureCameraExpansion() {
    const cards = Array.from(document.querySelectorAll(
        ".camera-card[data-camera]",
    ));
    const collapseCurrent = () => {
        if (!dashboardState.expandedCamera) return;
        dashboardState.expandedCamera.classList.remove("is-expanded");
        const hint = dashboardState.expandedCamera.querySelector(
            ".camera-zoom-hint",
        );
        if (hint) hint.textContent = "双击查看完整画面";
        dashboardState.expandedCamera = null;
        document.body.classList.remove("camera-image-expanded");
    };

    cards.forEach((card) => {
        const viewport = card.querySelector(".camera-viewport");
        if (!viewport) return;
        viewport.addEventListener("dblclick", (event) => {
            event.preventDefault();
            if (dashboardState.expandedCamera === card) {
                collapseCurrent();
                return;
            }
            collapseCurrent();
            card.classList.add("is-expanded");
            dashboardState.expandedCamera = card;
            document.body.classList.add("camera-image-expanded");
            const hint = card.querySelector(".camera-zoom-hint");
            if (hint) hint.textContent = "双击恢复原位置";
        });
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") collapseCurrent();
    });
}


function initialize() {
    configureCameraExpansion();
    configureMapHeading();
    configurePoolBoundary();
    configureVisualHistoryControls();
    configureBaseTrajectoryControl();
    configureMapInteraction();
    window.addEventListener("resize", () => {
        if (dashboardState.status) drawNavigation(dashboardState.status);
    });
    refreshStatus();
    window.setInterval(refreshStatus, 250);
}


initialize();
