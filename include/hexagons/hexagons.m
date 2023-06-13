% Generates a hexagon plot that represents the chosen basic incident directions in beam steering.
% radius: the radius of circumscribed circle of each polygon.
% Different radius should be used for different Gaussian beam waist used in subregion simulations.
% For a chosen waist w, radius is recommended to be no greater than sin(0.4 / (pi * w)).
% Running the following code generates the provided hexagon_2.5um.txt file:
% hexInfo = hexagons(0.0495);
% writematrix(hexInfo, "hexagon_2.5um.txt");
% The file specifies 499 basic incident directions that work for w = 2.5um in subregion simulations.
% DO NOT modify the following functions--they generate data of the right format.

function hexInfo = hexagons(radius)
    hold all;
    xlim([-1.2 1.2]);
    ylim([-1.2 1.2]);
    drawPolygon(0, 0, 1, 1000);
    [centers, ~] = drawCluster(radius, 0, 0, [], []);
    set(gca, 'Color', [0.8 0.8 0.8]);
    hexInfo = processHexagons(centers, radius);
end

function [centers, record] = drawCluster(radius, x0, y0, centers, record)
    cutoff = 1.0;
    record = [record; x0 y0];
    % Draw the center hexagon
    if ~findRepetition(x0, y0, centers) && x0 * x0 + y0 * y0 <= cutoff
        drawPolygon(x0, y0, radius, 6);
        centers = [centers; x0 y0];
    end
    % Draw neighbor 1
    x1 = x0;
    y1 = y0 + sqrt(3) * radius;
    if ~findRepetition(x1, y1, centers) && x1 * x1 + y1 * y1 <= cutoff
        drawPolygon(x1, y1, radius, 6);
        centers = [centers; x1 y1];
    end
    % Draw neighbor 2
    x2 = x0 + 1.5 * radius;
    y2 = y0 + sqrt(3) / 2 * radius;
    if ~findRepetition(x2, y2, centers) && x2 * x2 + y2 * y2 <= cutoff
        drawPolygon(x2, y2, radius, 6);
        centers = [centers; x2 y2];
    end
    % Draw neighbor 3
    x3 = x0 + 1.5 * radius;
    y3 = y0 - sqrt(3) / 2 * radius;
    if ~findRepetition(x3, y3, centers) && x3 * x3 + y3 * y3 <= cutoff
        drawPolygon(x3, y3, radius, 6);
        centers = [centers; x3 y3];
    end
    % Draw neighbor 4
    x4 = x0;
    y4 = y0 - sqrt(3) * radius;
    if ~findRepetition(x4, y4, centers) && x4 * x4 + y4 * y4 <= cutoff
        drawPolygon(x4, y4, radius, 6);
        centers = [centers; x4 y4];
    end
    % Draw neighbor 5
    x5 = x0 - 1.5 * radius;
    y5 = y0 - sqrt(3) / 2 * radius;
    if ~findRepetition(x5, y5, centers) && x5 * x5 + y5 * y5 <= cutoff
        drawPolygon(x5, y5, radius, 6);
        centers = [centers; x5 y5];
    end
    % Draw neighbor 6
    x6 = x0 - 1.5 * radius;
    y6 = y0 + sqrt(3) / 2 * radius;
    if ~findRepetition(x6, y6, centers) && x6 * x6 + y6 * y6 <= cutoff
        drawPolygon(x6, y6, radius, 6);
        centers = [centers; x6 y6];
    end
    % Recursion
    if ~findRepetition(x1, y1, record) && x1 * x1 + y1 * y1 <= cutoff
        [centers, record] = drawCluster(radius, x1, y1, centers, record);
    end
    if ~findRepetition(x2, y2, record) && x2 * x2 + y2 * y2 <= cutoff
        [centers, record] = drawCluster(radius, x2, y2, centers, record);
    end
    if ~findRepetition(x3, y3, record) && x3 * x3 + y3 * y3 <= cutoff
        [centers, record] = drawCluster(radius, x3, y3, centers, record);
    end
    if ~findRepetition(x4, y4, record) && x4 * x4 + y4 * y4 <= cutoff
        [centers, record] = drawCluster(radius, x4, y4, centers, record);
    end
    if ~findRepetition(x5, y5, record) && x5 * x5 + y5 * y5 <= cutoff
        [centers, record] = drawCluster(radius, x5, y5, centers, record);
    end
    if ~findRepetition(x6, y6, record) && x6 * x6 + y6 * y6 <= cutoff
        [centers, record] = drawCluster(radius, x6, y6, centers, record);
    end
end

function drawPolygon(x0, y0, scale, N_sides)
    t = (1 / (N_sides * 2) : 1 / N_sides : 1)' * 2 * pi;
    x = sin(t);
    y = cos(t);
    x = scale * [x; x(1)];
    y = scale * [y; y(1)];
    plot(x + x0, y + y0);
    if N_sides > 10
        color = [1 1 1];
    else
        color = 0.4 + 0.6 * [rand rand rand];
    end
    fill(x + x0, y + y0, color);
    axis square;
end

function found = findRepetition(x0, y0, centers)
    found = false;
    n = size(centers, 1);
    for i = 1:n
        x = centers(i, 1);
        y = centers(i, 2);
        if abs(x - x0) <= 0.001 && abs(y - y0) <= 0.001
            found  = true;
            break;
        end
    end
end

function info = processHexagons(centers, radius)
    n = size(centers, 1);
    info = zeros(n, 4);
    info(:, 1:2) = centers;
    for i = 1:n
        xc = centers(i, 1);
        yc = centers(i, 2);
        dist = sqrt(xc * xc + yc * yc);
        d1 = dist - radius;
        d2 = dist + radius;
        if d1 <= 0
            info(i, 3) = asin(dist);
        else
            t1 = asin(d1);
            if d2 <= sin(80 / 180 * pi)
                t2 = asin(d2);
            else
                t2 = 80 / 180 * pi;
            end
            info(i, 3) = (t1 + t2) / 2;
        end
        if xc == 0
            if yc == 0
                phi = 0;
            elseif yc > 0
                phi = pi / 2;
            else
                phi = pi * 3 / 2;
            end
        elseif yc == 0
            if (xc > 0)
                phi = 0;
            else
                phi = pi;
            end
        elseif xc > 0 && yc > 0
            phi = atan(yc / xc);
        elseif xc < 0 && yc > 0
            phi = atan(yc / xc) + pi;
        elseif xc < 0 && yc < 0
            phi = atan(yc / xc) + pi;
        else
            phi = atan(yc / xc) + 2 * pi;
        end
        info(i, 4) = phi;
    end
end